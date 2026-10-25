import logging
import uuid
from pathlib import PurePosixPath

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ConflictError, NotFoundError, QuotaExceededError
from app.models import Document, DocumentStatus, JobType, User
from app.repositories.document_repository import DocumentRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.file_validation import ValidatedUpload, clean_display_name, validate_upload
from app.services.storage import StorageBackend

logger = logging.getLogger(__name__)

NOT_FOUND = "Document not found."


class DocumentService:
    def __init__(
        self,
        db: AsyncSession,
        documents: DocumentRepository,
        jobs: ProcessingJobRepository,
        storage: StorageBackend,
        settings: Settings,
    ) -> None:
        self.db = db
        self.documents = documents
        self.jobs = jobs
        self.storage = storage
        self.settings = settings

    async def upload(
        self,
        *,
        user: User,
        filename: str | None,
        name: str | None,
        data: bytes,
        idempotency_key: str | None,
    ) -> tuple[Document, bool]:
        """Returns (document, created). created=False means an idempotent replay."""
        validated = validate_upload(filename, data, self.settings)
        display_name = (
            clean_display_name(name)
            or PurePosixPath(validated.original_filename).stem[:255]
            or "Untitled"
        )

        if idempotency_key:
            existing = await self.documents.get_by_idempotency_key(user.id, idempotency_key)
            if existing is not None:
                return self._replay(existing, validated), False

        if idempotency_key:
            existing = await self.documents.get_by_idempotency_key(user.id, idempotency_key)
            if existing is not None:
                return self._replay(existing, validated), False

        # A replay (above) is never blocked by the quota. Only genuinely new documents are.
        if await self.documents.count_for_user(user.id) >= self.settings.max_documents_per_user:
            raise QuotaExceededError(
                f"You have reached the limit of {self.settings.max_documents_per_user} "
                "documents. Delete some to upload more."
            )

        document_id = uuid.uuid4()

        document_id = uuid.uuid4()
        storage_key = f"{user.id}/{document_id}.{validated.file_type}"
        await self.storage.save(storage_key, validated.data)
        try:
            try:
                # Savepoint: a unique violation rolls back only this insert.
                async with self.db.begin_nested():
                    document = await self.documents.create(
                        id=document_id,
                        user_id=user.id,
                        name=display_name,
                        original_filename=validated.original_filename,
                        file_type=validated.file_type,
                        file_size=validated.size,
                        content_sha256=validated.sha256,
                        storage_key=storage_key,
                        idempotency_key=idempotency_key,
                    )
            except IntegrityError:
                # A concurrent request with the same Idempotency-Key won the race.
                await self._remove_file(storage_key)
                existing = (
                    await self.documents.get_by_idempotency_key(user.id, idempotency_key)
                    if idempotency_key
                    else None
                )
                if existing is None:
                    raise
                return self._replay(existing, validated), False

            await self.jobs.enqueue(document.id, JobType.PROCESS)
            await self.db.commit()  # document + job become visible together, or not at all
        except Exception:
            await self.db.rollback()
            await self._remove_file(storage_key)
            raise
        return document, True

    async def get(self, user: User, document_id: uuid.UUID) -> Document:
        document = await self.documents.get_owned(document_id, user.id)
        if document is None:
            raise NotFoundError(NOT_FOUND)
        return document

    async def list(
        self, user: User, *, limit: int, offset: int, status: DocumentStatus | None
    ) -> tuple[list[Document], int]:
        return await self.documents.list_owned(user.id, limit=limit, offset=offset, status=status)

    async def request_delete(self, user: User, document_id: uuid.UUID) -> Document:
        """Hide the document immediately and queue the cleanup. Safe to retry."""
        document = await self.documents.get_owned(document_id, user.id, for_update=True)
        if document is None:
            raise NotFoundError(NOT_FOUND)
        document.status = DocumentStatus.DELETING
        document.error_message = None
        await self.jobs.enqueue(document.id, JobType.DELETE)
        await self.db.commit()
        return document

    # ---- internals --------------------------------------------------------------------

    @staticmethod
    def _replay(existing: Document, validated: ValidatedUpload) -> Document:
        if existing.status == DocumentStatus.DELETING:
            raise ConflictError("That upload was deleted. Use a new Idempotency-Key.")
        if existing.content_sha256 != validated.sha256:
            raise ConflictError("This Idempotency-Key was already used with a different file.")
        return existing

    async def _remove_file(self, storage_key: str) -> None:
        try:
            await self.storage.delete(storage_key)
        except Exception:
            logger.warning("Could not remove stored file %s", storage_key, exc_info=True)
