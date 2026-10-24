import uuid

import pytest
from sqlalchemy import select, update

from app.models import Document, DocumentStatus, JobType, ProcessingJob
from tests.helpers import SOME_ID, files_on_disk

DOCS = "/api/v1/documents"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", DOCS),
        ("GET", f"{DOCS}/{SOME_ID}"),
        ("GET", f"{DOCS}/{SOME_ID}/status"),
        ("DELETE", f"{DOCS}/{SOME_ID}"),
    ],
)
async def test_endpoints_require_authentication(api_client, method, path):
    resp = await api_client.request(method, path)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_list_is_empty_for_new_user(api_client, alice):
    resp = await api_client.get(DOCS, headers=alice)

    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_list_paginates_newest_first(api_client, upload, alice):
    ids = []
    for i in range(5):
        created = await upload(alice, filename=f"doc{i}.txt", content=f"content {i}".encode())
        ids.append(created.json()["id"])

    first = (await api_client.get(DOCS, params={"limit": 2}, headers=alice)).json()
    last = (await api_client.get(DOCS, params={"limit": 2, "offset": 4}, headers=alice)).json()
    beyond = (await api_client.get(DOCS, params={"offset": 10}, headers=alice)).json()

    assert first["total"] == 5
    assert first["limit"] == 2
    assert [d["id"] for d in first["items"]] == [ids[4], ids[3]]
    assert [d["id"] for d in last["items"]] == [ids[0]]
    assert beyond["items"] == []
    assert beyond["total"] == 5


@pytest.mark.parametrize(("param", "value"), [("limit", 0), ("limit", 101), ("offset", -1)])
async def test_list_rejects_out_of_range_pagination(api_client, alice, param, value):
    resp = await api_client.get(DOCS, params={param: value}, headers=alice)

    assert resp.status_code == 422


async def test_list_filters_by_status(api_client, upload, alice, session_factory):
    await upload(alice, content=b"one")
    second = await upload(alice, content=b"two")
    async with session_factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == uuid.UUID(second.json()["id"]))
            .values(status=DocumentStatus.FAILED, error_message="could not parse")
        )
        await session.commit()

    resp = await api_client.get(DOCS, params={"status": "failed"}, headers=alice)

    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == second.json()["id"]
    assert body["items"][0]["error_message"] == "could not parse"


async def test_list_only_returns_own_documents(api_client, upload, alice, bob):
    await upload(alice, content=b"alice 1")
    await upload(alice, content=b"alice 2")
    await upload(bob, content=b"bob 1")

    alice_list = (await api_client.get(DOCS, headers=alice)).json()
    bob_list = (await api_client.get(DOCS, headers=bob)).json()

    assert alice_list["total"] == 2
    assert bob_list["total"] == 1
    assert {d["id"] for d in alice_list["items"]}.isdisjoint({d["id"] for d in bob_list["items"]})


async def test_owner_can_get_details_and_status(api_client, upload, alice):
    doc_id = (await upload(alice)).json()["id"]

    detail = await api_client.get(f"{DOCS}/{doc_id}", headers=alice)
    status = await api_client.get(f"{DOCS}/{doc_id}/status", headers=alice)

    assert detail.status_code == 200
    assert detail.json()["id"] == doc_id
    assert status.status_code == 200
    assert set(status.json()) == {"id", "status", "chunk_count", "error_message", "updated_at"}
    assert status.json()["status"] == "pending"


@pytest.mark.parametrize(("method", "suffix"), [("GET", ""), ("GET", "/status"), ("DELETE", "")])
async def test_other_users_document_is_indistinguishable_from_missing(
    api_client, upload, alice, bob, method, suffix
):
    doc_id = (await upload(alice)).json()["id"]

    foreign = await api_client.request(method, f"{DOCS}/{doc_id}{suffix}", headers=bob)
    missing = await api_client.request(method, f"{DOCS}/{uuid.uuid4()}{suffix}", headers=bob)

    assert foreign.status_code == missing.status_code == 404
    foreign_error, missing_error = foreign.json()["error"], missing.json()["error"]
    foreign_error.pop("request_id")
    missing_error.pop("request_id")
    assert foreign_error == missing_error
    # Alice's document is untouched by Bob's attempt.
    mine = await api_client.get(f"{DOCS}/{doc_id}", headers=alice)
    assert mine.status_code == 200
    assert mine.json()["status"] == "pending"


async def test_malformed_document_id_is_422(api_client, alice):
    resp = await api_client.get(f"{DOCS}/not-a-uuid", headers=alice)

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


async def test_delete_hides_document_and_queues_cleanup(
    api_client, upload, alice, session_factory, storage
):
    doc_id = (await upload(alice)).json()["id"]

    resp = await api_client.delete(f"{DOCS}/{doc_id}", headers=alice)

    assert resp.status_code == 202
    assert resp.json()["status"] == "deleting"
    assert (await api_client.get(f"{DOCS}/{doc_id}", headers=alice)).status_code == 404
    assert (await api_client.get(f"{DOCS}/{doc_id}/status", headers=alice)).status_code == 404
    listing = (await api_client.get(DOCS, headers=alice)).json()
    assert listing["items"] == []
    assert listing["total"] == 0
    async with session_factory() as session:
        job_types = (
            (
                await session.execute(
                    select(ProcessingJob.job_type).order_by(ProcessingJob.created_at)
                )
            )
            .scalars()
            .all()
        )
    assert job_types == [JobType.PROCESS, JobType.DELETE]
    assert len(files_on_disk(storage)) == 1  # the worker removes it in Phase 5


async def test_deleting_twice_returns_404(api_client, upload, alice, session_factory):
    doc_id = (await upload(alice)).json()["id"]
    await api_client.delete(f"{DOCS}/{doc_id}", headers=alice)

    again = await api_client.delete(f"{DOCS}/{doc_id}", headers=alice)

    assert again.status_code == 404
    async with session_factory() as session:
        deletes = (
            (
                await session.execute(
                    select(ProcessingJob).where(ProcessingJob.job_type == JobType.DELETE)
                )
            )
            .scalars()
            .all()
        )
    assert len(deletes) == 1  # no duplicate cleanup job
