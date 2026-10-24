import asyncio
import hashlib

from sqlalchemy import func, select

from app.core.config import get_settings
from app.models import Document, DocumentStatus, JobStatus, JobType, ProcessingJob
from app.repositories.processing_job_repository import ProcessingJobRepository
from tests.helpers import files_on_disk

DOCS = "/api/v1/documents"
PUBLIC_FIELDS = {
    "id", "name", "original_filename", "file_type", "file_size", "status",
    "chunk_count", "error_message", "created_at", "updated_at",
}  # fmt: skip
PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF"


async def _count(session_factory, model) -> int:
    async with session_factory() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_upload_requires_authentication(api_client):
    resp = await api_client.post(DOCS, files={"file": ("a.txt", b"hello", "text/plain")})

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_upload_txt_creates_pending_document_and_job(upload, alice, session_factory, storage):
    resp = await upload(alice, filename="notes.txt", content=b"hello world", name="My notes")

    assert resp.status_code == 202
    body = resp.json()
    assert set(body) == PUBLIC_FIELDS  # no storage_key or hash leaks
    assert body["name"] == "My notes"
    assert body["original_filename"] == "notes.txt"
    assert body["file_type"] == "txt"
    assert body["file_size"] == 11
    assert body["status"] == "pending"
    assert body["chunk_count"] == 0
    assert body["error_message"] is None
    assert resp.headers["location"] == f"{DOCS}/{body['id']}"

    async with session_factory() as session:
        doc = (await session.execute(select(Document))).scalar_one()
        job = (await session.execute(select(ProcessingJob))).scalar_one()
    assert str(doc.id) == body["id"]
    assert doc.status == DocumentStatus.PENDING
    assert doc.content_sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert doc.storage_key == f"{doc.user_id}/{doc.id}.txt"
    assert (storage.root / doc.storage_key).read_bytes() == b"hello world"
    assert job.document_id == doc.id
    assert job.job_type == JobType.PROCESS
    assert job.status == JobStatus.QUEUED


async def test_name_defaults_to_filename_without_extension(upload, alice):
    resp = await upload(alice, filename="Quarterly Report.txt")

    assert resp.status_code == 202
    assert resp.json()["name"] == "Quarterly Report"


async def test_upload_accepts_pdf(upload, alice):
    resp = await upload(alice, filename="paper.pdf", content=PDF)

    assert resp.status_code == 202
    assert resp.json()["file_type"] == "pdf"


async def test_upload_rejects_content_that_does_not_match_type(upload, alice, storage):
    resp = await upload(alice, filename="fake.pdf", content=b"just some text")

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"
    assert files_on_disk(storage) == []


async def test_upload_rejects_unsupported_type(upload, alice):
    resp = await upload(alice, filename="program.exe", content=b"MZ\x90\x00")

    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


async def test_upload_rejects_empty_file(upload, alice):
    resp = await upload(alice, content=b"")

    assert resp.status_code == 400


async def test_upload_without_file_is_422(api_client, alice):
    resp = await api_client.post(DOCS, data={"name": "no file here"}, headers=alice)

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


async def test_oversized_upload_is_rejected_and_nothing_is_stored(
    api_app, test_settings, upload, alice, session_factory, storage
):
    small = test_settings.model_copy(update={"max_upload_mb": 1})
    api_app.dependency_overrides[get_settings] = lambda: small

    resp = await upload(alice, content=b"a" * (1024 * 1024 + 1))

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
    assert files_on_disk(storage) == []
    assert await _count(session_factory, Document) == 0


async def test_hostile_filename_cannot_escape_storage(upload, alice, storage):
    resp = await upload(alice, filename="../../etc/passwd.txt")

    assert resp.status_code == 202
    body = resp.json()
    assert body["original_filename"] == "passwd.txt"
    (stored,) = files_on_disk(storage)
    assert stored.resolve().is_relative_to(storage.root)
    assert stored.name == f"{body['id']}.txt"


# ---- idempotency --------------------------------------------------------------------


async def test_retry_with_same_key_returns_original_document(
    upload, alice, session_factory, storage
):
    first = await upload(alice, key="retry-key-0001")
    second = await upload(alice, key="retry-key-0001")

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert await _count(session_factory, Document) == 1
    assert await _count(session_factory, ProcessingJob) == 1
    assert len(files_on_disk(storage)) == 1


async def test_same_key_with_different_file_is_conflict(upload, alice, session_factory):
    await upload(alice, key="retry-key-0002", content=b"first version")

    resp = await upload(alice, key="retry-key-0002", content=b"second version")

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "conflict"
    assert await _count(session_factory, Document) == 1


async def test_idempotency_keys_are_scoped_per_user(upload, alice, bob):
    first = await upload(alice, key="shared-key-0001")
    second = await upload(bob, key="shared-key-0001")

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] != second.json()["id"]


async def test_concurrent_uploads_with_same_key_create_one_document(
    upload, alice, session_factory, storage
):
    results = await asyncio.gather(
        upload(alice, key="race-key-0001"), upload(alice, key="race-key-0001")
    )

    assert sorted(r.status_code for r in results) == [200, 202]
    assert len({r.json()["id"] for r in results}) == 1
    assert await _count(session_factory, Document) == 1
    assert len(files_on_disk(storage)) == 1  # the loser removed its orphan file


async def test_invalid_idempotency_key_is_422(upload, alice):
    resp = await upload(alice, key="short")

    assert resp.status_code == 422


# ---- failure handling ---------------------------------------------------------------


async def test_failure_after_file_save_removes_file_and_row(
    upload, alice, session_factory, storage, monkeypatch
):
    async def boom(self, document_id, job_type):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(ProcessingJobRepository, "enqueue", boom)

    resp = await upload(alice)

    assert resp.status_code == 500
    assert files_on_disk(storage) == []
    assert await _count(session_factory, Document) == 0
