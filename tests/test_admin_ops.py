import uuid

import pytest
from sqlalchemy import delete, func, select, update

from app.models import (
    Document,
    DocumentStatus,
    JobType,
    ProcessingJob,
    RefreshToken,
    User,
)
from app.workers.admin_ops import finalize_user_deletion, request_reindex, request_user_deletion


async def set_status(session_factory, doc_id, status) -> None:
    async with session_factory() as db:
        await db.execute(update(Document).where(Document.id == doc_id).values(status=status))
        await db.commit()


async def upload_three(upload, headers) -> list[uuid.UUID]:
    return [
        uuid.UUID((await upload(headers, content=f"doc {i}".encode())).json()["id"])
        for i in range(3)
    ]


async def process_job_owners(session_factory) -> list[uuid.UUID]:
    async with session_factory() as db:
        rows = await db.execute(
            select(ProcessingJob.document_id).where(ProcessingJob.job_type == JobType.PROCESS)
        )
        return list(rows.scalars())


# ---- reindex ------------------------------------------------------------------------


async def test_reindex_requeues_only_completed_documents(upload, alice, session_factory):
    completed, failed, deleting = await upload_three(upload, alice)
    await set_status(session_factory, completed, DocumentStatus.COMPLETED)
    await set_status(session_factory, failed, DocumentStatus.FAILED)
    await set_status(session_factory, deleting, DocumentStatus.DELETING)

    assert await request_reindex(session_factory) == 1

    async with session_factory() as db:
        statuses = {d.id: d.status for d in (await db.execute(select(Document))).scalars()}
    assert statuses == {
        completed: DocumentStatus.PENDING,
        failed: DocumentStatus.FAILED,
        deleting: DocumentStatus.DELETING,
    }
    owners = await process_job_owners(session_factory)
    assert owners.count(completed) == 2  # the original upload job plus the new one
    assert owners.count(failed) == 1


async def test_reindex_dry_run_changes_nothing(upload, alice, session_factory):
    (doc_id, *_) = await upload_three(upload, alice)
    await set_status(session_factory, doc_id, DocumentStatus.COMPLETED)

    assert await request_reindex(session_factory, dry_run=True) == 1

    async with session_factory() as db:
        assert (await db.get(Document, doc_id)).status == DocumentStatus.COMPLETED
    assert (await process_job_owners(session_factory)).count(doc_id) == 1


async def test_reindexing_twice_does_not_queue_duplicates(upload, alice, session_factory):
    (doc_id, *_) = await upload_three(upload, alice)
    await set_status(session_factory, doc_id, DocumentStatus.COMPLETED)
    await request_reindex(session_factory)

    assert await request_reindex(session_factory) == 0  # it is pending now

    assert (await process_job_owners(session_factory)).count(doc_id) == 2


# ---- account deletion -----------------------------------------------------------------


async def user_by_email(session_factory, email: str) -> User | None:
    async with session_factory() as db:
        return (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()


async def test_deletion_request_hides_data_and_locks_the_account(
    upload, alice, bob, session_factory
):
    mine = [(await upload(alice, content=f"a{i}".encode())).json()["id"] for i in range(2)]
    theirs = (await upload(bob)).json()["id"]

    assert await request_user_deletion(session_factory, "Alice@Example.com") == 2

    async with session_factory() as db:
        alice_user = (
            await db.execute(select(User).where(User.email == "alice@example.com"))
        ).scalar_one()
        bob_user = (
            await db.execute(select(User).where(User.email == "bob@example.com"))
        ).scalar_one()
        docs = {str(d.id): d.status for d in (await db.execute(select(Document))).scalars()}
        delete_jobs = (
            (
                await db.execute(
                    select(ProcessingJob.document_id).where(
                        ProcessingJob.job_type == JobType.DELETE
                    )
                )
            )
            .scalars()
            .all()
        )
        live_tokens = (
            await db.execute(
                select(func.count())
                .select_from(RefreshToken)
                .where(RefreshToken.user_id == alice_user.id, RefreshToken.revoked_at.is_(None))
            )
        ).scalar_one()
    assert not alice_user.is_active
    assert bob_user.is_active
    assert {docs[doc_id] for doc_id in mine} == {DocumentStatus.DELETING}
    assert docs[theirs] == DocumentStatus.PENDING
    assert sorted(str(d) for d in delete_jobs) == sorted(mine)
    assert live_tokens == 0


async def test_a_locked_account_is_rejected_immediately(api_client, upload, alice, session_factory):
    await upload(alice)
    await request_user_deletion(session_factory, "alice@example.com")

    resp = await api_client.get("/api/v1/users/me", headers=alice)

    assert resp.status_code == 401


async def test_finalize_waits_while_documents_remain(upload, alice, session_factory):
    await upload(alice)
    await request_user_deletion(session_factory, "alice@example.com")

    assert await finalize_user_deletion(session_factory, "alice@example.com") is False
    assert await user_by_email(session_factory, "alice@example.com") is not None


async def test_finalize_deletes_the_user_once_the_worker_has_removed_the_documents(
    upload, alice, bob, session_factory
):
    await upload(alice)
    await upload(bob)
    await request_user_deletion(session_factory, "alice@example.com")
    alice_user = await user_by_email(session_factory, "alice@example.com")
    async with session_factory() as db:  # what the worker's delete jobs do
        await db.execute(delete(Document).where(Document.user_id == alice_user.id))
        await db.commit()

    assert await finalize_user_deletion(session_factory, "alice@example.com") is True

    assert await user_by_email(session_factory, "alice@example.com") is None
    assert await user_by_email(session_factory, "bob@example.com") is not None


async def test_finalize_refuses_an_active_account(alice, session_factory):
    with pytest.raises(ValueError):
        await finalize_user_deletion(session_factory, "alice@example.com")


async def test_unknown_users_are_reported(session_factory):
    with pytest.raises(LookupError):
        await request_user_deletion(session_factory, "nobody@example.com")
    with pytest.raises(LookupError):
        await finalize_user_deletion(session_factory, "nobody@example.com")
