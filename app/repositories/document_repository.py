import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentStatus


class DocumentRepository:
    """Every read that serves a user filters by user_id inside the SQL itself."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        id: uuid.UUID,  # noqa: A002
        user_id: uuid.UUID,
        name: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        content_sha256: str,
        storage_key: str,
        idempotency_key: str | None,
    ) -> Document:
        document = Document(
            id=id,
            user_id=user_id,
            name=name,
            original_filename=original_filename,
            file_type=file_type,
            file_size=file_size,
            content_sha256=content_sha256,
            storage_key=storage_key,
            status=DocumentStatus.PENDING,
            chunk_count=0,
            embedding_model=None,
            error_message=None,
            idempotency_key=idempotency_key,
        )
        self.db.add(document)
        await self.db.flush()  # surfaces unique-constraint violations immediately
        return document

    async def get_owned(
        self, document_id: uuid.UUID, user_id: uuid.UUID, *, for_update: bool = False
    ) -> Document | None:
        """Returns None for missing, foreign and being-deleted documents alike."""
        stmt = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.status != DocumentStatus.DELETING,
        )
        if for_update:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_by_idempotency_key(self, user_id: uuid.UUID, key: str) -> Document | None:
        stmt = select(Document).where(Document.user_id == user_id, Document.idempotency_key == key)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def list_owned(
        self,
        user_id: uuid.UUID,
        *,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        conditions = [Document.user_id == user_id, Document.status != DocumentStatus.DELETING]
        if status is not None:
            conditions.append(Document.status == status)

        total = (
            await self.db.execute(select(func.count()).select_from(Document).where(*conditions))
        ).scalar_one()
        rows = await self.db.execute(
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows.scalars().all()), total
