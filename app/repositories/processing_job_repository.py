import uuid
from datetime import UTC, datetime

from sqlalchemy import and_, exists, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models import JobStatus, JobType, ProcessingJob

MAX_ATTEMPTS = {JobType.PROCESS: 5, JobType.DELETE: 10}


class ProcessingJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def enqueue(self, document_id: uuid.UUID, job_type: JobType) -> ProcessingJob:
        job = ProcessingJob(
            document_id=document_id, job_type=job_type, max_attempts=MAX_ATTEMPTS[job_type]
        )
        self.db.add(job)
        await self.db.flush()
        return job

    async def claim_next(self, worker_id: str, now: datetime) -> ProcessingJob | None:
        """Atomically take the oldest due job. SKIP LOCKED lets many workers poll at once
        without blocking each other or ever claiming the same row. The caller commits.

        A delete job is not claimable while a process job for the same document is running,
        so vectors can never be re-inserted after the delete removed them."""
        running_process = aliased(ProcessingJob)
        blocked_delete = and_(
            ProcessingJob.job_type == JobType.DELETE,
            exists().where(
                running_process.document_id == ProcessingJob.document_id,
                running_process.job_type == JobType.PROCESS,
                running_process.status == JobStatus.RUNNING,
            ),
        )
        stmt = (
            select(ProcessingJob)
            .where(
                ProcessingJob.status == JobStatus.QUEUED,
                ProcessingJob.run_after <= now,
                ~blocked_delete,
            )
            .order_by(ProcessingJob.run_after, ProcessingJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
        job = (await self.db.execute(stmt)).scalar_one_or_none()
        if job is None:
            return None
        job.status = JobStatus.RUNNING
        job.attempts += 1  # counted at claim time, so a crashing job can't retry forever
        job.locked_at = now
        job.locked_by = worker_id
        await self.db.flush()
        return job

    async def heartbeat(self, job_id: uuid.UUID) -> None:
        await self._update_running(job_id, locked_at=datetime.now(UTC))

    async def mark_succeeded(self, job_id: uuid.UUID) -> None:
        await self._update_running(
            job_id, status=JobStatus.SUCCEEDED, locked_at=None, locked_by=None, last_error=None
        )

    async def mark_failed(self, job_id: uuid.UUID, error: str) -> None:
        await self._update_running(
            job_id, status=JobStatus.FAILED, locked_at=None, locked_by=None, last_error=error
        )

    async def reschedule(self, job_id: uuid.UUID, *, run_after: datetime, error: str) -> None:
        await self._update_running(
            job_id,
            status=JobStatus.QUEUED,
            run_after=run_after,
            locked_at=None,
            locked_by=None,
            last_error=error,
        )

    async def release(self, job_id: uuid.UUID) -> None:
        """Graceful shutdown: hand the job back without using up an attempt."""
        await self._update_running(
            job_id,
            status=JobStatus.QUEUED,
            attempts=func.greatest(ProcessingJob.attempts - 1, 0),
            run_after=datetime.now(UTC),
            locked_at=None,
            locked_by=None,
        )

    async def find_stale(self, cutoff: datetime) -> list[ProcessingJob]:
        stmt = (
            select(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.RUNNING, ProcessingJob.locked_at < cutoff)
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def _update_running(self, job_id: uuid.UUID, **values: object) -> None:
        # The RUNNING guard stops a slow worker from overwriting a job that was already
        # recovered and handed to someone else.
        await self.db.execute(
            update(ProcessingJob)
            .where(ProcessingJob.id == job_id, ProcessingJob.status == JobStatus.RUNNING)
            .values(**values)
        )
