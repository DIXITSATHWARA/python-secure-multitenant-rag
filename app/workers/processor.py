"""Executes claimed jobs. Holds no database connection while calling OpenAI or Qdrant."""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.models import DocumentStatus, JobType, ProcessingJob
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.chunking import split_text
from app.services.embedding_service import EmbeddingService
from app.services.errors import GENERIC_PROCESSING_FAILURE, PermanentProcessingError
from app.services.storage import StorageBackend
from app.services.text_extraction import extract_text
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    document_id: uuid.UUID
    job_type: JobType
    attempts: int
    max_attempts: int

    @classmethod
    def from_model(cls, job: ProcessingJob) -> Self:
        return cls(job.id, job.document_id, job.job_type, job.attempts, job.max_attempts)


@dataclass(frozen=True)
class _DocumentSnapshot:
    id: uuid.UUID
    user_id: uuid.UUID
    file_type: str
    storage_key: str | None


def compute_backoff(attempts: int, base: int, cap: int) -> float:
    """Seconds to wait before the next attempt: base, 2x, 4x ... capped."""
    return float(min(base * 2 ** max(attempts - 1, 0), cap))


async def claim_job(session_factory: SessionFactory, worker_id: str) -> ClaimedJob | None:
    async with session_factory() as db:
        job = await ProcessingJobRepository(db).claim_next(worker_id, datetime.now(UTC))
        if job is None:
            return None
        claimed = ClaimedJob.from_model(job)
        await db.commit()
    return claimed


class DocumentProcessor:
    def __init__(
        self,
        session_factory: SessionFactory,
        storage: StorageBackend,
        vectors: VectorService,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._vectors = vectors
        self._embeddings = embeddings
        self._settings = settings

    # ---- entry point ------------------------------------------------------------------

    async def run(self, job: ClaimedJob) -> None:
        """Run one claimed job and record its outcome."""
        try:
            if job.job_type == JobType.PROCESS:
                await self._process(job)
            else:
                await self._delete(job)
        except asyncio.CancelledError:
            await asyncio.shield(self._release(job))  # shutting down: give the job back
            raise
        except PermanentProcessingError as exc:
            await self._fail_permanently(job, exc.message)
        except Exception as exc:
            await self._retry_or_fail(job, exc)
        else:
            async with self._session_factory() as db:
                await ProcessingJobRepository(db).mark_succeeded(job.id)
                await db.commit()

    # ---- process job ------------------------------------------------------------------

    async def _process(self, job: ClaimedJob) -> None:
        snapshot = await self._begin_processing(job.document_id)
        if snapshot is None:
            return

        data = await self._read_file(snapshot)
        text = await extract_text(snapshot.file_type, data, self._settings)
        chunks = await asyncio.to_thread(
            split_text, text, self._settings.chunk_size, self._settings.chunk_overlap
        )
        if not chunks:
            raise PermanentProcessingError("No readable text was found in this document.")
        if len(chunks) > self._settings.max_chunks_per_document:
            raise PermanentProcessingError(
                "This document is too large to index "
                f"(over {self._settings.max_chunks_per_document} sections)."
            )

        await self._vectors.ensure_collection()
        # Clean slate: makes a retry after a partial failure idempotent, and removes stale
        # points if the chunking settings changed between attempts.
        await self._vectors.delete_document(user_id=snapshot.user_id, document_id=snapshot.id)

        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embedded = await self._embeddings.embed_texts([c.text for c in batch])
            await self._vectors.upsert_chunks(
                user_id=snapshot.user_id,
                document_id=snapshot.id,
                embedding_model=self._settings.openai_embedding_model,
                chunks=batch,
                vectors=embedded,
            )
            status = await self._checkpoint(job.id, snapshot.id)
            if status is None:  # the document row is gone (e.g. the account was deleted)
                await self._cleanup_orphan_vectors(snapshot)
                return
            if status == DocumentStatus.DELETING:
                # Stop spending money. The delete job runs after this job ends and cleans up.
                logger.info("Document %s is being deleted; stopping early", snapshot.id)
                return

        stored = await self._vectors.count_document_points(
            user_id=snapshot.user_id, document_id=snapshot.id
        )
        if stored != len(chunks):
            raise ExternalServiceError("Stored vector count does not match; retrying.")
        await self._finish_processing(snapshot, len(chunks))

    async def _begin_processing(self, document_id: uuid.UUID) -> _DocumentSnapshot | None:
        async with self._session_factory() as db:
            doc = await DocumentRepository(db).get_for_worker(document_id, for_update=True)
            if doc is None or doc.status in (DocumentStatus.DELETING, DocumentStatus.COMPLETED):
                logger.info(
                    "Skipping process job for document %s (missing, deleting or done)", document_id
                )
                return None
            doc.status = DocumentStatus.PROCESSING
            doc.error_message = None
            snapshot = _DocumentSnapshot(doc.id, doc.user_id, doc.file_type, doc.storage_key)
            await db.commit()
        return snapshot

    async def _read_file(self, snapshot: _DocumentSnapshot) -> bytes:
        if not snapshot.storage_key:
            raise PermanentProcessingError("The original file is missing. Please upload it again.")
        try:
            return await self._storage.read(snapshot.storage_key)
        except FileNotFoundError:
            raise PermanentProcessingError(
                "The uploaded file could not be found. Please upload it again."
            ) from None

    async def _checkpoint(self, job_id: uuid.UUID, document_id: uuid.UUID) -> DocumentStatus | None:
        """Renew the lease and report the document's current status."""
        async with self._session_factory() as db:
            await ProcessingJobRepository(db).heartbeat(job_id)
            status = await DocumentRepository(db).get_status(document_id)
            await db.commit()
        return status

    async def _finish_processing(self, snapshot: _DocumentSnapshot, chunk_count: int) -> None:
        orphaned = False
        async with self._session_factory() as db:
            # Row lock: serializes with a concurrent delete request.
            doc = await DocumentRepository(db).get_for_worker(snapshot.id, for_update=True)
            if doc is None:
                orphaned = True
            elif doc.status == DocumentStatus.DELETING:
                logger.info("Document %s was deleted while processing; not completing", doc.id)
            else:
                doc.status = DocumentStatus.COMPLETED
                doc.chunk_count = chunk_count
                doc.embedding_model = self._settings.openai_embedding_model
                doc.error_message = None
                await db.commit()
                logger.info("Document %s completed (%d chunks)", doc.id, chunk_count)
        if orphaned:
            await self._cleanup_orphan_vectors(snapshot)

    async def _cleanup_orphan_vectors(self, snapshot: _DocumentSnapshot) -> None:
        try:
            await self._vectors.delete_document(user_id=snapshot.user_id, document_id=snapshot.id)
        except ExternalServiceError:
            logger.warning("Could not remove orphan vectors for document %s", snapshot.id)

    # ---- delete job -------------------------------------------------------------------

    async def _delete(self, job: ClaimedJob) -> None:
        async with self._session_factory() as db:
            doc = await DocumentRepository(db).get_for_worker(job.document_id)
            if doc is None:
                return  # already gone
            if doc.status != DocumentStatus.DELETING:
                logger.warning(
                    "Delete job for document %s, which is not 'deleting'; ignoring", doc.id
                )
                return
            snapshot = _DocumentSnapshot(doc.id, doc.user_id, doc.file_type, doc.storage_key)

        # Order matters: vectors, then file, then the row. Every step is idempotent, so a
        # crash anywhere just means the job retries and converges.
        await self._vectors.delete_document(user_id=snapshot.user_id, document_id=snapshot.id)
        if snapshot.storage_key:
            await self._storage.delete(snapshot.storage_key)
        async with self._session_factory() as db:
            await DocumentRepository(db).delete_if_deleting(snapshot.id)  # cascades to jobs
            await db.commit()
        logger.info("Document %s deleted", snapshot.id)

    # ---- failure policy ---------------------------------------------------------------

    async def _fail_permanently(self, job: ClaimedJob, message: str) -> None:
        logger.warning("Job %s failed permanently: %s", job.id, message)
        async with self._session_factory() as db:
            if job.job_type == JobType.PROCESS:
                await DocumentRepository(db).mark_failed(job.document_id, message)
            await ProcessingJobRepository(db).mark_failed(job.id, message)
            await db.commit()

    async def _retry_or_fail(self, job: ClaimedJob, exc: Exception) -> None:
        # last_error is internal (never returned by the API) but still kept short.
        error_text = f"{type(exc).__name__}: {exc}"[:500]
        if isinstance(exc, ExternalServiceError):
            logger.warning(
                "Job %s attempt %d/%d failed: %s", job.id, job.attempts, job.max_attempts, exc
            )
        else:
            logger.error(
                "Job %s attempt %d/%d crashed", job.id, job.attempts, job.max_attempts, exc_info=exc
            )

        async with self._session_factory() as db:
            jobs = ProcessingJobRepository(db)
            if job.attempts >= job.max_attempts:
                if job.job_type == JobType.PROCESS:
                    await DocumentRepository(db).mark_failed(
                        job.document_id, GENERIC_PROCESSING_FAILURE
                    )
                else:
                    logger.error(
                        "Delete job %s gave up; document %s stays 'deleting' until an operator "
                        "runs scripts.requeue_jobs",
                        job.id,
                        job.document_id,
                    )
                await jobs.mark_failed(job.id, error_text)
            else:
                delay = compute_backoff(
                    job.attempts,
                    self._settings.job_retry_base_seconds,
                    self._settings.job_retry_max_seconds,
                )
                delay += random.uniform(0, delay * 0.1)  # noqa: S311  (jitter, not security)
                await jobs.reschedule(
                    job.id, run_after=datetime.now(UTC) + timedelta(seconds=delay), error=error_text
                )
            await db.commit()

    async def _release(self, job: ClaimedJob) -> None:
        try:
            async with self._session_factory() as db:
                await ProcessingJobRepository(db).release(job.id)
                await db.commit()
        except Exception:
            logger.warning("Could not release job %s on shutdown", job.id, exc_info=True)
