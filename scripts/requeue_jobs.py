"""Requeue failed jobs and make waiting jobs due now.  Run: python -m scripts.requeue_jobs"""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models import JobStatus, ProcessingJob


async def main() -> None:
    engine = create_async_engine(get_settings().database_url)
    now = datetime.now(UTC)
    try:
        async with engine.begin() as conn:
            failed = await conn.execute(
                update(ProcessingJob)
                .where(ProcessingJob.status == JobStatus.FAILED)
                .values(status=JobStatus.QUEUED, attempts=0, run_after=now, last_error=None)
            )
            waiting = await conn.execute(
                update(ProcessingJob)
                .where(ProcessingJob.status == JobStatus.QUEUED, ProcessingJob.run_after > now)
                .values(run_after=now)
            )
        print(
            f"Requeued {failed.rowcount} failed job(s); made {waiting.rowcount} waiting job(s) due."
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
