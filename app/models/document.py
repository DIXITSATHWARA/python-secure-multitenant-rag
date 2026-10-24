import uuid

from sqlalchemy import BigInteger, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import DocumentStatus, string_enum
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(BigInteger)
    content_sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[DocumentStatus] = mapped_column(
        string_enum(DocumentStatus, "document_status"),
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )
    chunk_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    # Vectors from different embedding models are not comparable. Record which one was used.
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    # User-safe message only. Never store raw stack traces here.
    error_message: Mapped[str | None] = mapped_column(Text)
    # Client-supplied header that makes upload retries safe (Phase 4).
    idempotency_key: Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (
        # NULLs are distinct in Postgres, so documents without a key never collide.
        UniqueConstraint("user_id", "idempotency_key", name="uq_documents_user_id_idempotency_key"),
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
        Index("ix_documents_user_id_status", "user_id", "status"),
    )
