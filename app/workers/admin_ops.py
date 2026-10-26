"""Operator actions: re-embed documents and delete accounts safely."""

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Document, DocumentStatus, JobType, RefreshToken, User
from app.repositories.processing_job_repository import ProcessingJobRepository

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


async def request_reindex(session_factory: SessionFactory, *, dry_run: bool = False) -> int:
    """Re-queue processing for every completed document. Returns how many were (or would be).

    Documents go back to 'pending', so they are not searchable until the worker finishes them.
    Use it after losing the Qdrant collection, or after changing the embedding model, the
    collection name or the chunk settings."""
    async with session_factory() as db:
        result = await db.execute(
            select(Document).where(Document.status == DocumentStatus.COMPLETED).with_for_update()
        )
        documents = list(result.scalars())
        if dry_run:
            return len(documents)
        jobs = ProcessingJobRepository(db)
        for document in documents:
            document.status = DocumentStatus.PENDING
            await jobs.enqueue(document.id, JobType.PROCESS)
        await db.commit()
    return len(documents)


async def _get_user(db: AsyncSession, email: str) -> User:
    result = await db.execute(select(User).where(User.email == email.strip().lower()))
    user = result.scalar_one_or_none()
    if user is None:
        raise LookupError(f"No user with email {email!r}")
    return user


async def request_user_deletion(session_factory: SessionFactory, email: str) -> int:
    """Step 1: lock the account and queue cleanup of everything it owns. Returns the number
    of documents queued. The worker removes vectors and files, then each document row."""
    async with session_factory() as db:
        user = await _get_user(db, email)
        user.is_active = False  # access tokens stop working on the very next request
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
        result = await db.execute(
            select(Document)
            .where(Document.user_id == user.id, Document.status != DocumentStatus.DELETING)
            .with_for_update()
        )
        documents = list(result.scalars())
        jobs = ProcessingJobRepository(db)
        for document in documents:
            document.status = DocumentStatus.DELETING
            document.error_message = None
            await jobs.enqueue(document.id, JobType.DELETE)
        await db.commit()
    logger.info("Account deletion requested: %d document(s) queued", len(documents))
    return len(documents)


async def finalize_user_deletion(session_factory: SessionFactory, email: str) -> bool:
    """Step 2: delete the user row once the worker has removed every document.
    Returns False while documents remain. Refuses to touch an active account."""
    async with session_factory() as db:
        user = await _get_user(db, email)
        if user.is_active:
            raise ValueError("Run request_user_deletion first: the account is still active.")
        remaining = (
            await db.execute(
                select(func.count()).select_from(Document).where(Document.user_id == user.id)
            )
        ).scalar_one()
        if remaining:
            return False
        await db.delete(user)  # refresh tokens go with it (ON DELETE CASCADE)
        await db.commit()
    return True
