from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Document, DocumentStatus, JobStatus, ProcessingJob

SessionFactory = async_sessionmaker[AsyncSession]
STUCK_DELETING_AFTER = timedelta(hours=1)


@dataclass(frozen=True)
class OpsStats:
    queued: int
    running: int
    failed: int
    oldest_due_age_seconds: float | None  # how long the longest-waiting runnable job has waited
    stale_running: int  # running, but the worker stopped renewing the lease
    stuck_deleting: int  # documents stuck in "deleting" for over an hour

    def problems(self, *, max_queue_age_seconds: int = 600) -> list[str]:
        found: list[str] = []
        if self.failed:
            found.append(
                f"{self.failed} failed job(s): inspect last_error with scripts.show_documents, "
                "then run scripts.requeue_jobs"
            )
        age = self.oldest_due_age_seconds
        if age is not None and age > max_queue_age_seconds:
            found.append(f"the oldest runnable job has waited {age:.0f}s: is the worker running?")
        if self.stale_running:
            found.append(f"{self.stale_running} running job(s) have an expired lease")
        if self.stuck_deleting:
            found.append(f"{self.stuck_deleting} document(s) stuck in 'deleting' for over 1 hour")
        return found


async def collect_ops_stats(
    session_factory: SessionFactory, *, lease_seconds: int, now: datetime | None = None
) -> OpsStats:
    now = now or datetime.now(UTC)
    async with session_factory() as db:
        rows = await db.execute(
            select(ProcessingJob.status, func.count()).group_by(ProcessingJob.status)
        )
        counts = {row[0]: row[1] for row in rows}
        oldest = (
            await db.execute(
                select(func.min(ProcessingJob.run_after)).where(
                    ProcessingJob.status == JobStatus.QUEUED, ProcessingJob.run_after <= now
                )
            )
        ).scalar_one()
        stale = (
            await db.execute(
                select(func.count())
                .select_from(ProcessingJob)
                .where(
                    ProcessingJob.status == JobStatus.RUNNING,
                    ProcessingJob.locked_at < now - timedelta(seconds=lease_seconds),
                )
            )
        ).scalar_one()
        stuck = (
            await db.execute(
                select(func.count())
                .select_from(Document)
                .where(
                    Document.status == DocumentStatus.DELETING,
                    Document.updated_at < now - STUCK_DELETING_AFTER,
                )
            )
        ).scalar_one()
    return OpsStats(
        queued=counts.get(JobStatus.QUEUED, 0),
        running=counts.get(JobStatus.RUNNING, 0),
        failed=counts.get(JobStatus.FAILED, 0),
        oldest_due_age_seconds=(now - oldest).total_seconds() if oldest else None,
        stale_running=stale,
        stuck_deleting=stuck,
    )
