import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JobType, ProcessingJob


class ProcessingJobRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def enqueue(self, document_id: uuid.UUID, job_type: JobType) -> ProcessingJob:
        job = ProcessingJob(document_id=document_id, job_type=job_type)
        self.db.add(job)
        await self.db.flush()
        return job
