import os
import time
import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete, update

from app.core.config import get_settings
from app.models import Document, DocumentStatus
from app.services.chunking import Chunk
from app.services.vector_service import VectorService
from app.workers.sweep import SweepRefused, sweep_orphans
from tests.helpers import files_on_disk

settings = get_settings().model_copy(
    update={"openai_embedding_dimensions": 4, "qdrant_collection": "test_sweep"}
)
VECTOR = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def vectors(qdrant) -> VectorService:
    return VectorService(qdrant, settings)


def age(path, hours: float = 48) -> None:
    moment = time.time() - hours * 3600
    os.utime(path, (moment, moment))


def age_all(storage) -> None:
    for path in files_on_disk(storage):
        age(path)


async def make_doc(upload, headers, vectors, session_factory, content: bytes):
    """A document created through the real API, with one vector stored for it."""
    resp = await upload(headers, content=content)
    assert resp.status_code == 202, resp.text
    doc_id = uuid.UUID(resp.json()["id"])
    async with session_factory() as db:
        doc = await db.get(Document, doc_id)
        user_id, key = doc.user_id, doc.storage_key
    await vectors.upsert_chunks(
        user_id=user_id,
        document_id=doc_id,
        embedding_model="t",
        chunks=[Chunk(0, "text")],
        vectors=[VECTOR],
    )
    return doc_id, user_id, key


async def drop_row(session_factory, doc_id) -> None:
    """Simulates an account deleted straight from the database."""
    async with session_factory() as db:
        await db.execute(delete(Document).where(Document.id == doc_id))
        await db.commit()


async def test_dry_run_reports_orphans_but_deletes_nothing(
    upload, alice, vectors, session_factory, storage
):
    await make_doc(upload, alice, vectors, session_factory, b"kept")
    lost, lost_user, lost_key = await make_doc(upload, alice, vectors, session_factory, b"lost")
    await drop_row(session_factory, lost)
    age_all(storage)

    report = await sweep_orphans(session_factory, storage, vectors, apply=False)

    assert report.orphan_files == [lost_key]
    assert report.orphan_vector_documents == [(lost_user, lost)]
    assert report.files_deleted == report.vectors_deleted == 0
    assert len(files_on_disk(storage)) == 2
    assert await vectors.count_document_points(user_id=lost_user, document_id=lost) == 1


async def test_apply_removes_only_the_orphans(upload, alice, vectors, session_factory, storage):
    kept, kept_user, kept_key = await make_doc(upload, alice, vectors, session_factory, b"kept")
    lost, lost_user, _ = await make_doc(upload, alice, vectors, session_factory, b"lost")
    await drop_row(session_factory, lost)
    age_all(storage)

    report = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert (report.files_deleted, report.vectors_deleted, report.errors) == (1, 1, 0)
    assert [p.relative_to(storage.root).as_posix() for p in files_on_disk(storage)] == [kept_key]
    assert await vectors.count_document_points(user_id=lost_user, document_id=lost) == 0
    assert await vectors.count_document_points(user_id=kept_user, document_id=kept) == 1


async def test_recent_files_are_protected_by_the_grace_period(
    upload, alice, vectors, session_factory, storage
):
    await make_doc(upload, alice, vectors, session_factory, b"kept")
    lost, lost_user, _ = await make_doc(upload, alice, vectors, session_factory, b"lost")
    await drop_row(session_factory, lost)  # files are brand new: an upload may still be mid-flight

    report = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert report.orphan_files == []
    assert len(files_on_disk(storage)) == 2
    # Vectors always have a row first, so they have no grace period.
    assert report.orphan_vector_documents == [(lost_user, lost)]


async def test_documents_being_deleted_are_not_orphans(
    upload, alice, vectors, session_factory, storage
):
    doc_id, _, _ = await make_doc(upload, alice, vectors, session_factory, b"going")
    async with session_factory() as db:
        await db.execute(
            update(Document).where(Document.id == doc_id).values(status=DocumentStatus.DELETING)
        )
        await db.commit()
    age_all(storage)

    report = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert report.orphan_files == []
    assert report.orphan_vector_documents == []  # the delete job owns this cleanup
    assert len(files_on_disk(storage)) == 1


async def test_unrecognized_files_are_never_touched(vectors, session_factory, storage):
    (storage.root / "notes.txt").write_bytes(b"x")
    (storage.root / "random" / "dir").mkdir(parents=True)
    (storage.root / "random" / "dir" / "file.bin").write_bytes(b"y")
    age_all(storage)

    report = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert report.orphan_files == []
    assert len(files_on_disk(storage)) == 2


async def test_stale_tmp_files_are_removed(upload, alice, vectors, session_factory, storage):
    _, _, kept_key = await make_doc(upload, alice, vectors, session_factory, b"kept")
    tmp_key = f"{uuid.uuid4()}/{uuid.uuid4()}.txt.tmp"
    await storage.save(tmp_key, b"half written")
    age_all(storage)

    report = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert report.orphan_files == [tmp_key]
    assert report.files_deleted == 1
    assert [p.relative_to(storage.root).as_posix() for p in files_on_disk(storage)] == [kept_key]


async def test_refuses_to_delete_when_the_database_is_empty_unless_forced(
    vectors, session_factory, storage
):
    user, doc = uuid.uuid4(), uuid.uuid4()
    await vectors.upsert_chunks(
        user_id=user, document_id=doc, embedding_model="t", chunks=[Chunk(0, "x")], vectors=[VECTOR]
    )

    with pytest.raises(SweepRefused):
        await sweep_orphans(session_factory, storage, vectors, apply=True)
    assert await vectors.count_document_points(user_id=user, document_id=doc) == 1

    report = await sweep_orphans(session_factory, storage, vectors, apply=True, force=True)

    assert report.vectors_deleted == 1
    assert await vectors.count_document_points(user_id=user, document_id=doc) == 0


async def test_sweeping_twice_finds_nothing_the_second_time(
    upload, alice, vectors, session_factory, storage
):
    await make_doc(upload, alice, vectors, session_factory, b"kept")
    lost, _, _ = await make_doc(upload, alice, vectors, session_factory, b"lost")
    await drop_row(session_factory, lost)
    age_all(storage)
    await sweep_orphans(session_factory, storage, vectors, apply=True)

    second = await sweep_orphans(session_factory, storage, vectors, apply=True)

    assert second.orphan_files == []
    assert second.orphan_vector_documents == []
    assert (second.files_deleted, second.vectors_deleted) == (0, 0)


async def test_list_document_owners_pages_through_every_point(vectors, monkeypatch):
    monkeypatch.setattr("app.services.vector_service.SCROLL_PAGE_SIZE", 2)
    expected = {}
    for _ in range(3):
        user, doc = uuid.uuid4(), uuid.uuid4()
        await vectors.upsert_chunks(
            user_id=user,
            document_id=doc,
            embedding_model="t",
            chunks=[Chunk(0, "a"), Chunk(1, "b")],
            vectors=[VECTOR, VECTOR],
        )
        expected[doc] = user  # 6 points in total: three pages of two

    assert await vectors.list_document_owners() == expected
