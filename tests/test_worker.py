import asyncio
import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select, update

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.models import (
    Document,
    DocumentStatus,
    JobStatus,
    JobType,
    ProcessingJob,
    RefreshToken,
    User,
)
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.errors import GENERIC_PROCESSING_FAILURE
from app.services.vector_service import VectorService
from app.workers.maintenance import purge_old_rows, recover_stale_jobs
from app.workers.processor import DocumentProcessor, claim_job, compute_backoff
from tests.helpers import files_on_disk

DIMS = 8
COLLECTION = "test_chunks"
LONG_TEXT = ("Alpha beta gamma delta. " * 40 + "\n\n") * 3  # ~2.9k chars -> ~16 chunks of 200


class FakeEmbeddings:
    """Deterministic stand-in for OpenAI. Can be told to fail."""

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.calls = 0
        self.texts = 0
        self.error: Exception | None = None  # raised on every call while set
        self.fail_on: dict[int, Exception] = {}  # call number -> exception
        self.on_call = None  # async callable(call_number), runs before the embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.on_call is not None:
            await self.on_call(self.calls)
        if self.error is not None:
            raise self.error
        if self.calls in self.fail_on:
            raise self.fail_on[self.calls]
        self.texts += len(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [1.0 + digest[i] / 255 for i in range(self.dims)]  # never the zero vector


@pytest.fixture
def worker_settings(test_settings: Settings) -> Settings:
    return test_settings.model_copy(
        update={
            "openai_embedding_dimensions": DIMS,
            "qdrant_collection": COLLECTION,
            "chunk_size": 200,
            "chunk_overlap": 20,
            "embedding_batch_size": 2,
            "max_chunks_per_document": 50,
        }
    )


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def vectors(qdrant, worker_settings) -> VectorService:
    return VectorService(qdrant, worker_settings)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(DIMS)


@pytest.fixture
def processor(session_factory, storage, vectors, embeddings, worker_settings) -> DocumentProcessor:
    return DocumentProcessor(session_factory, storage, vectors, embeddings, worker_settings)


# ---- helpers ------------------------------------------------------------------------


async def _upload_doc(upload, headers, **kwargs) -> uuid.UUID:
    resp = await upload(headers, **kwargs)
    assert resp.status_code == 202, resp.text
    return uuid.UUID(resp.json()["id"])


async def run_next(session_factory, processor):
    job = await claim_job(session_factory, "test-worker")
    if job is not None:
        await processor.run(job)
    return job


async def drain(session_factory, processor) -> int:
    """Run every currently-due job. Returns how many ran."""
    ran = 0
    while await run_next(session_factory, processor) is not None:
        ran += 1
        assert ran < 20, "worker loop did not settle"
    return ran


async def get_doc(session_factory, doc_id) -> Document | None:
    async with session_factory() as db:
        return await db.get(Document, doc_id)


async def get_jobs(session_factory, doc_id) -> list[ProcessingJob]:
    async with session_factory() as db:
        rows = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == doc_id)
            .order_by(ProcessingJob.created_at)
        )
        return list(rows.scalars())


async def point_count(qdrant, name: str = COLLECTION) -> int:
    if not await qdrant.collection_exists(name):
        return 0
    return (await qdrant.count(name, exact=True)).count


async def make_jobs_due(session_factory) -> None:
    async with session_factory() as db:
        await db.execute(
            update(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.QUEUED)
            .values(run_after=datetime.now(UTC) - timedelta(seconds=1))
        )
        await db.commit()


# ---- processing ---------------------------------------------------------------------


async def test_process_job_embeds_chunks_and_completes(
    upload, alice, session_factory, processor, embeddings, qdrant, worker_settings
):
    doc_id = await _upload_doc(
        upload,
        alice,
        filename="Secret Plan.txt",
        content=LONG_TEXT.encode(),
        name="Top secret name",
    )

    ran = await drain(session_factory, processor)

    assert ran == 1
    doc = await get_doc(session_factory, doc_id)
    assert doc.status == DocumentStatus.COMPLETED
    assert doc.chunk_count > 2
    assert doc.embedding_model == worker_settings.openai_embedding_model
    assert doc.error_message is None
    assert embeddings.texts == doc.chunk_count
    assert await point_count(qdrant) == doc.chunk_count
    assert [j.status for j in await get_jobs(session_factory, doc_id)] == [JobStatus.SUCCEEDED]

    points, _ = await qdrant.scroll(COLLECTION, limit=100, with_payload=True)
    for point in points:
        # Only these fields: no document name, no filename, no email.
        assert set(point.payload) == {
            "user_id",
            "document_id",
            "chunk_index",
            "text",
            "embedding_model",
        }
        assert point.payload["document_id"] == str(doc_id)
        assert point.payload["user_id"] == str(doc.user_id)
    assert {p.payload["chunk_index"] for p in points} == set(range(doc.chunk_count))


async def test_completed_document_is_not_reprocessed(
    upload, alice, session_factory, processor, embeddings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await drain(session_factory, processor)
    calls_before = embeddings.calls
    async with session_factory() as db:  # e.g. a duplicate job from a crash-and-retry
        await ProcessingJobRepository(db).enqueue(doc_id, JobType.PROCESS)
        await db.commit()

    await drain(session_factory, processor)

    assert embeddings.calls == calls_before


async def test_reprocessing_does_not_duplicate_vectors(
    upload, alice, session_factory, processor, qdrant
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await drain(session_factory, processor)
    first_count = await point_count(qdrant)
    async with session_factory() as db:
        await db.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(status=DocumentStatus.PENDING, chunk_count=0)
        )
        await ProcessingJobRepository(db).enqueue(doc_id, JobType.PROCESS)
        await db.commit()

    await drain(session_factory, processor)

    assert await point_count(qdrant) == first_count
    assert (await get_doc(session_factory, doc_id)).status == DocumentStatus.COMPLETED


async def test_transient_failure_is_retried_and_partial_vectors_are_replaced(
    upload, alice, session_factory, processor, embeddings, qdrant
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    embeddings.fail_on = {2: ExternalServiceError("rate limited")}  # first batch succeeds

    ran = await drain(session_factory, processor)

    assert ran == 1  # rescheduled into the future, so not claimable yet
    doc = await get_doc(session_factory, doc_id)
    (job,) = await get_jobs(session_factory, doc_id)
    assert doc.status == DocumentStatus.PROCESSING
    assert job.status == JobStatus.QUEUED
    assert job.attempts == 1
    assert job.last_error.startswith("ExternalServiceError")
    assert job.run_after > datetime.now(UTC) + timedelta(seconds=20)  # backoff applied
    assert await point_count(qdrant) == 2  # the partial write from the failed attempt

    embeddings.fail_on = {}
    await make_jobs_due(session_factory)
    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    (job,) = await get_jobs(session_factory, doc_id)
    assert doc.status == DocumentStatus.COMPLETED
    assert job.status == JobStatus.SUCCEEDED
    assert job.attempts == 2
    assert await point_count(qdrant) == doc.chunk_count  # exactly one point per chunk


async def test_document_fails_after_attempts_are_exhausted(
    upload, alice, session_factory, processor, embeddings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    async with session_factory() as db:
        await db.execute(update(ProcessingJob).values(max_attempts=1))
        await db.commit()
    embeddings.error = ExternalServiceError("OpenAI is down")

    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    (job,) = await get_jobs(session_factory, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message == GENERIC_PROCESSING_FAILURE
    assert job.status == JobStatus.FAILED


async def test_corrupt_pdf_fails_permanently_without_retries(
    upload, alice, session_factory, processor, embeddings
):
    doc_id = await _upload_doc(
        upload, alice, filename="bad.pdf", content=b"%PDF-1.4 this is not really a pdf"
    )

    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    (job,) = await get_jobs(session_factory, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message
    assert job.status == JobStatus.FAILED
    assert job.attempts == 1
    assert embeddings.calls == 0


async def test_blank_text_file_fails_with_a_helpful_message(
    upload, alice, session_factory, processor
):
    doc_id = await _upload_doc(upload, alice, content=b"   \n\n  \t ")

    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert "No readable text" in doc.error_message


async def test_document_with_too_many_chunks_is_rejected(
    upload, alice, session_factory, storage, vectors, embeddings, worker_settings
):
    small = worker_settings.model_copy(update={"max_chunks_per_document": 2})
    limited = DocumentProcessor(session_factory, storage, vectors, embeddings, small)
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())

    await drain(session_factory, limited)

    doc = await get_doc(session_factory, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert "too large" in doc.error_message
    assert embeddings.calls == 0


async def test_missing_original_file_fails_permanently(
    upload, alice, session_factory, processor, storage
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    for stored in files_on_disk(storage):
        stored.unlink()

    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert "could not be found" in doc.error_message


async def test_deletion_during_processing_stops_early(
    upload, alice, session_factory, processor, embeddings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())

    async def request_delete(call_number: int) -> None:
        if call_number == 1:
            async with session_factory() as db:
                await db.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(status=DocumentStatus.DELETING)
                )
                await db.commit()

    embeddings.on_call = request_delete

    await drain(session_factory, processor)

    doc = await get_doc(session_factory, doc_id)
    (job,) = await get_jobs(session_factory, doc_id)
    assert doc.status == DocumentStatus.DELETING  # never flipped to completed
    assert job.status == JobStatus.SUCCEEDED
    assert embeddings.calls == 1  # no more money spent after the delete request


# ---- deletion -----------------------------------------------------------------------


async def test_delete_removes_only_that_documents_data(
    api_client, upload, alice, bob, session_factory, processor, qdrant, storage
):
    alice_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    bob_id = await _upload_doc(upload, bob, content=LONG_TEXT.encode())
    await drain(session_factory, processor)
    bob_doc = await get_doc(session_factory, bob_id)
    assert await point_count(qdrant) == 2 * bob_doc.chunk_count

    resp = await api_client.delete(f"/api/v1/documents/{alice_id}", headers=alice)
    assert resp.status_code == 202
    await drain(session_factory, processor)

    assert await get_doc(session_factory, alice_id) is None
    assert await get_jobs(session_factory, alice_id) == []  # cascaded
    assert await point_count(qdrant) == bob_doc.chunk_count
    points, _ = await qdrant.scroll(COLLECTION, limit=100, with_payload=True)
    assert {p.payload["user_id"] for p in points} == {str(bob_doc.user_id)}
    (remaining,) = files_on_disk(storage)
    assert remaining.name.startswith(str(bob_id))
    assert (await get_doc(session_factory, bob_id)).status == DocumentStatus.COMPLETED


async def test_delete_before_processing_skips_embedding_and_cleans_up(
    api_client, upload, alice, session_factory, processor, embeddings, qdrant, storage
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await api_client.delete(f"/api/v1/documents/{doc_id}", headers=alice)

    await drain(session_factory, processor)

    assert embeddings.calls == 0
    assert await get_doc(session_factory, doc_id) is None
    assert files_on_disk(storage) == []
    assert await point_count(qdrant) == 0


async def test_delete_job_waits_while_a_process_job_is_running(
    api_client, upload, alice, session_factory
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await api_client.delete(f"/api/v1/documents/{doc_id}", headers=alice)  # process + delete queued
    now = datetime.now(UTC)

    async with session_factory() as db:  # a worker claims the process job and is "mid-flight"
        first = await ProcessingJobRepository(db).claim_next("worker-1", now)
        await db.commit()
    assert first.job_type == JobType.PROCESS

    async with session_factory() as db:  # the delete job must not be claimable yet
        assert await ProcessingJobRepository(db).claim_next("worker-2", now) is None

    async with session_factory() as db:
        await ProcessingJobRepository(db).mark_succeeded(first.id)
        await db.commit()

    async with session_factory() as db:
        second = await ProcessingJobRepository(db).claim_next("worker-2", now)
        await db.commit()
    assert second is not None
    assert second.job_type == JobType.DELETE


async def test_failed_vector_delete_is_retried_until_it_succeeds(
    api_client, upload, alice, session_factory, processor, vectors, monkeypatch
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await drain(session_factory, processor)  # processed normally
    real_delete = vectors.delete_document
    calls = {"n": 0}

    async def flaky_delete(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ExternalServiceError("Qdrant is down")
        await real_delete(**kwargs)

    monkeypatch.setattr(vectors, "delete_document", flaky_delete)
    await api_client.delete(f"/api/v1/documents/{doc_id}", headers=alice)

    await drain(session_factory, processor)  # first attempt fails and is rescheduled

    doc = await get_doc(session_factory, doc_id)
    assert doc is not None
    assert doc.status == DocumentStatus.DELETING  # still hidden from the user, not lost
    assert (await get_jobs(session_factory, doc_id))[-1].status == JobStatus.QUEUED

    await make_jobs_due(session_factory)
    await drain(session_factory, processor)

    assert await get_doc(session_factory, doc_id) is None
    assert calls["n"] == 2


# ---- crash recovery, queue behavior, maintenance --------------------------------------


async def test_stale_job_is_requeued_and_then_completes(
    upload, alice, session_factory, processor, worker_settings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    claimed = await claim_job(session_factory, "dead-worker")
    assert claimed is not None
    assert claimed.attempts == 1
    async with session_factory() as db:  # the worker died two hours ago
        await db.execute(
            update(ProcessingJob).values(locked_at=datetime.now(UTC) - timedelta(hours=2))
        )
        await db.commit()

    assert await recover_stale_jobs(session_factory, worker_settings) == 1

    (job,) = await get_jobs(session_factory, doc_id)
    assert job.status == JobStatus.QUEUED
    assert job.locked_at is None
    assert "lease" in job.last_error
    await drain(session_factory, processor)
    assert (await get_doc(session_factory, doc_id)).status == DocumentStatus.COMPLETED


async def test_stale_job_out_of_attempts_fails_the_document(
    upload, alice, session_factory, worker_settings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await claim_job(session_factory, "dead-worker")
    async with session_factory() as db:
        await db.execute(
            update(ProcessingJob).values(
                attempts=5, locked_at=datetime.now(UTC) - timedelta(hours=2)
            )
        )
        await db.commit()

    assert await recover_stale_jobs(session_factory, worker_settings) == 1

    (job,) = await get_jobs(session_factory, doc_id)
    doc = await get_doc(session_factory, doc_id)
    assert job.status == JobStatus.FAILED
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message == GENERIC_PROCESSING_FAILURE


async def test_running_job_with_a_fresh_lease_is_left_alone(
    upload, alice, session_factory, worker_settings
):
    doc_id = await _upload_doc(upload, alice, content=LONG_TEXT.encode())
    await claim_job(session_factory, "busy-worker")

    assert await recover_stale_jobs(session_factory, worker_settings) == 0

    (job,) = await get_jobs(session_factory, doc_id)
    assert job.status == JobStatus.RUNNING


async def test_concurrent_workers_never_claim_the_same_job(upload, alice, session_factory):
    await _upload_doc(upload, alice, content=b"first document")
    await _upload_doc(upload, alice, content=b"second document")

    async def claim(worker_id: str) -> uuid.UUID:
        async with session_factory() as db:
            job = await ProcessingJobRepository(db).claim_next(worker_id, datetime.now(UTC))
            await asyncio.sleep(0.2)  # hold the row lock so the other worker must skip it
            await db.commit()
            return job.id

    first, second = await asyncio.gather(claim("a"), claim("b"))

    assert first != second


async def test_purge_removes_only_old_rows(upload, alice, session_factory):
    doc_id = await _upload_doc(upload, alice)  # leaves one queued job
    now = datetime.now(UTC)
    async with session_factory() as db:
        user_id = (await db.execute(select(User.id))).scalar_one()
        db.add_all(
            [
                RefreshToken(
                    user_id=user_id,
                    family_id=uuid.uuid4(),
                    token_hash="a" * 64,
                    expires_at=now - timedelta(days=2),  # purged
                ),
                RefreshToken(
                    user_id=user_id,
                    family_id=uuid.uuid4(),
                    token_hash="b" * 64,
                    expires_at=now - timedelta(hours=1),  # kept: within the 1-day grace
                ),
                RefreshToken(
                    user_id=user_id,
                    family_id=uuid.uuid4(),
                    token_hash="c" * 64,
                    expires_at=now + timedelta(days=5),  # kept: still valid
                ),
                ProcessingJob(
                    document_id=doc_id,
                    job_type=JobType.PROCESS,
                    status=JobStatus.SUCCEEDED,
                    updated_at=now - timedelta(days=10),  # purged
                ),
                ProcessingJob(
                    document_id=doc_id, job_type=JobType.PROCESS, status=JobStatus.SUCCEEDED
                ),  # kept: recent
            ]
        )
        await db.commit()

    assert await purge_old_rows(session_factory) == (1, 1)

    async with session_factory() as db:
        tokens = (await db.execute(select(func.count()).select_from(RefreshToken))).scalar_one()
        jobs = (await db.execute(select(func.count()).select_from(ProcessingJob))).scalar_one()
    assert tokens == 3  # Alice's own login token plus the two kept above
    assert jobs == 2  # the queued upload job plus the recent succeeded job


def test_backoff_doubles_and_is_capped():
    assert compute_backoff(1, 30, 900) == 30
    assert compute_backoff(2, 30, 900) == 60
    assert compute_backoff(5, 30, 900) == 480
    assert compute_backoff(6, 30, 900) == 900
    assert compute_backoff(20, 30, 900) == 900
