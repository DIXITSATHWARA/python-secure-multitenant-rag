import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import JobStatus, JobType, string_enum
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin, utcnow


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Durable work queue. The worker claims rows with SELECT ... FOR UPDATE SKIP LOCKED."""

    __tablename__ = "processing_jobs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[JobType] = mapped_column(string_enum(JobType, "job_type"))
    status: Mapped[JobStatus] = mapped_column(
        string_enum(JobStatus, "job_status"),
        default=JobStatus.QUEUED,
        server_default=JobStatus.QUEUED.value,
    )
    attempts: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(default=5, server_default=text("5"))
    # Retry backoff: on failure the worker pushes run_after into the future.
    run_after: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    locked_at: Mapped[datetime | None] = mapped_column()
    locked_by: Mapped[str | None] = mapped_column(String(100))
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_processing_jobs_status_run_after", "status", "run_after"),)
