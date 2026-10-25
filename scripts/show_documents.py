"""Print documents and queued jobs.  Run: python -m scripts.show_documents"""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models import Document, ProcessingJob


async def main() -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            docs = await conn.execute(
                select(
                    Document.id,
                    Document.name,
                    Document.status,
                    Document.chunk_count,
                    Document.error_message,
                ).order_by(Document.created_at)
            )
            print("\ndocuments")
            for d in docs:
                print(f"  {d.id}  {d.status!s:<10} chunks={d.chunk_count:<4} {d.name}")
                if d.error_message:
                    print(f"      error: {d.error_message}")
            jobs = await conn.execute(
                select(
                    ProcessingJob.document_id,
                    ProcessingJob.job_type,
                    ProcessingJob.status,
                    ProcessingJob.attempts,
                    ProcessingJob.max_attempts,
                    ProcessingJob.run_after,
                    ProcessingJob.last_error,
                ).order_by(ProcessingJob.created_at)
            )
            print("\nprocessing_jobs")
            for j in jobs:
                print(
                    f"  {j.document_id}  {j.job_type!s:<8} {j.status!s:<10} "
                    f"attempts={j.attempts}/{j.max_attempts}  run_after={j.run_after:%H:%M:%S}"
                )
                if j.last_error:
                    print(f"      last_error: {j.last_error}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
