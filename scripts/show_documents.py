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
                select(Document.id, Document.name, Document.status, Document.file_size).order_by(
                    Document.created_at
                )
            )
            print("\ndocuments")
            for d in docs:
                print(f"  {d.id}  {d.status!s:<10} {d.file_size:>8} B  {d.name}")
            jobs = await conn.execute(
                select(
                    ProcessingJob.document_id,
                    ProcessingJob.job_type,
                    ProcessingJob.status,
                    ProcessingJob.attempts,
                ).order_by(ProcessingJob.created_at)
            )
            print("\nprocessing_jobs")
            for j in jobs:
                print(
                    f"  {j.document_id}  {j.job_type!s:<8} {j.status!s:<10} attempts={j.attempts}"
                )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
