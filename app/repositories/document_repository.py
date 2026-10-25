import uuid
from collections.abc import Collection

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentStatus


class DocumentRepository:
    """User-facing reads filter by user_id inside the SQL itself. The worker-only methods
    at the bottom are NOT scoped to a user and must never be called from API code."""

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

    async def count_for_user(self, user_id: uuid.UUID) -> int:
        """All of the user's documents, including ones still being deleted."""
        stmt = select(func.count()).select_from(Document).where(Document.user_id == user_id)
        return (await self.db.execute(stmt)).scalar_one()

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

    async def get_completed_names(
        self, user_id: uuid.UUID, document_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """{id: name} for those of `document_ids` that belong to the user AND are completed."""
        if not document_ids:
            return {}
        stmt = select(Document.id, Document.name).where(
            Document.user_id == user_id,
            Document.id.in_(list(document_ids)),
            Document.status == DocumentStatus.COMPLETED,
        )
        return {row.id: row.name for row in await self.db.execute(stmt)}

    # ---- worker-only (not scoped to a user) -------------------------------------------

    async def get_for_worker(
        self, document_id: uuid.UUID, *, for_update: bool = False
    ) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        if for_update:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_status(self, document_id: uuid.UUID) -> DocumentStatus | None:
        stmt = select(Document.status).where(Document.id == document_id)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def mark_failed(self, document_id: uuid.UUID, message: str) -> None:
        """Never resurrects a document that is being deleted."""
        await self.db.execute(
            update(Document)
            .where(Document.id == document_id, Document.status != DocumentStatus.DELETING)
            .values(status=DocumentStatus.FAILED, error_message=message[:1000])
        )

    async def delete_if_deleting(self, document_id: uuid.UUID) -> None:
        """Removes the row (and, via ON DELETE CASCADE, its jobs). The status guard means a
        live document can never be deleted by a stray job."""
        await self.db.execute(
            delete(Document).where(
                Document.id == document_id, Document.status == DocumentStatus.DELETING
            )
        )
