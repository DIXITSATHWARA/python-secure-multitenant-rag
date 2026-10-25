import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models import JobStatus, JobType, ProcessingJob, RefreshToken
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.errors import GENERIC_PROCESSING_FAILURE

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]
SUCCEEDED_JOB_RETENTION_DAYS = 7
EXPIRED_TOKEN_GRACE_DAYS = 1


async def recover_stale_jobs(session_factory: SessionFactory, settings: Settings) -> int:
    """Re-queue jobs whose worker died (lease expired). Jobs out of attempts are failed."""
    now = datetime.now(UTC)
    cutoff = now - timedelta(seconds=settings.job_lease_seconds)
    async with session_factory() as db:
        stale = await ProcessingJobRepository(db).find_stale(cutoff)
        documents = DocumentRepository(db)
        for job in stale:
            job.locked_at = None
            job.locked_by = None
            if job.attempts >= job.max_attempts:
                job.status = JobStatus.FAILED
                job.last_error = "Worker lease expired and no attempts remain."
                if job.job_type == JobType.PROCESS:
                    await documents.mark_failed(job.document_id, GENERIC_PROCESSING_FAILURE)
                else:
                    logger.error("Delete job %s exhausted its attempts", job.id)
            else:
                job.status = JobStatus.QUEUED
                job.run_after = now
                job.last_error = "Worker lease expired; job requeued."
        await db.commit()
    return len(stale)


async def purge_old_rows(session_factory: SessionFactory) -> tuple[int, int]:
    """Delete long-expired refresh tokens and old succeeded jobs. Returns (tokens, jobs)."""
    now = datetime.now(UTC)
    async with session_factory() as db:
        tokens = await db.execute(
            delete(RefreshToken).where(
                RefreshToken.expires_at < now - timedelta(days=EXPIRED_TOKEN_GRACE_DAYS)
            )
        )
        jobs = await db.execute(
            delete(ProcessingJob).where(
                ProcessingJob.status == JobStatus.SUCCEEDED,
                ProcessingJob.updated_at < now - timedelta(days=SUCCEEDED_JOB_RETENTION_DAYS),
            )
        )
        await db.commit()
    return tokens.rowcount, jobs.rowcount
