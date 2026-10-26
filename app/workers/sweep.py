"""Find, and optionally remove, data that PostgreSQL no longer knows about.

Postgres is the source of truth. Files and vectors that no document row points to are
orphans, left behind by crashes between steps, or by an account that was deleted straight
from the database. The sweep only ever touches keys shaped like {uuid}/{uuid}.ext, and
protects recent files, because an upload writes its file *before* it inserts its row.
"""

import logging
import re
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ExternalServiceError
from app.models import Document
from app.services.storage import StorageBackend
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
MANAGED_KEY = re.compile(rf"^{_UUID}/{_UUID}\.[a-z0-9]{{2,5}}(\.tmp)?$")
QUERY_CHUNK = 500


class SweepRefused(RuntimeError):
    """The sweep stopped itself because the situation looks like a configuration mistake."""


@dataclass
class SweepReport:
    orphan_files: list[str] = field(default_factory=list)
    orphan_vector_documents: list[tuple[uuid.UUID, uuid.UUID]] = field(default_factory=list)
    files_deleted: int = 0
    vectors_deleted: int = 0
    errors: int = 0


def _chunks[T](items: Sequence[T], size: int = QUERY_CHUNK) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _known_storage_keys(db: AsyncSession, keys: Sequence[str]) -> set[str]:
    known: set[str] = set()
    for part in _chunks(keys):
        rows = await db.execute(select(Document.storage_key).where(Document.storage_key.in_(part)))
        known.update(key for key in rows.scalars() if key)
    return known


async def _existing_document_ids(db: AsyncSession, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
    existing: set[uuid.UUID] = set()
    for part in _chunks(ids):
        rows = await db.execute(select(Document.id).where(Document.id.in_(part)))
        existing.update(rows.scalars())
    return existing


async def sweep_orphans(
    session_factory: SessionFactory,
    storage: StorageBackend,
    vectors: VectorService,
    *,
    apply: bool,
    force: bool = False,
    grace: timedelta = timedelta(hours=24),
    now: datetime | None = None,
) -> SweepReport:
    now = now or datetime.now(UTC)
    report = SweepReport()

    # Snapshot storage and Qdrant FIRST, then ask Postgres. A document's row always exists
    # before its file's grace period ends and before its vectors are written, so anything
    # listed here without a row really is an orphan (or is already being cleaned up).
    objects = await storage.list_keys()
    candidates = sorted(
        o.key for o in objects if o.modified < now - grace and MANAGED_KEY.match(o.key)
    )
    owners = await vectors.list_document_owners()

    async with session_factory() as db:
        known_keys = await _known_storage_keys(db, candidates)
        existing_ids = await _existing_document_ids(db, list(owners))
        total_documents = (
            await db.execute(select(func.count()).select_from(Document))
        ).scalar_one()

    report.orphan_files = [key for key in candidates if key not in known_keys]
    report.orphan_vector_documents = sorted(
        ((owner, doc) for doc, owner in owners.items() if doc not in existing_ids),
        key=lambda pair: (str(pair[0]), str(pair[1])),
    )

    found = bool(report.orphan_files or report.orphan_vector_documents)
    if apply and found and total_documents == 0 and not force:
        raise SweepRefused(
            "The documents table is empty, but storage or Qdrant still hold data. That usually "
            "means DATABASE_URL points at the wrong database. Re-run with --force only if you "
            "are sure (for example after deleting the last account)."
        )
    if not apply:
        return report

    for key in report.orphan_files:
        try:
            await storage.delete(key)
            report.files_deleted += 1
        except Exception:
            logger.warning("Could not delete orphan file %s", key, exc_info=True)
            report.errors += 1
    for user_id, document_id in report.orphan_vector_documents:
        try:
            await vectors.delete_document(user_id=user_id, document_id=document_id)
            report.vectors_deleted += 1
        except ExternalServiceError:
            logger.warning("Could not delete orphan vectors of document %s", document_id)
            report.errors += 1
    return report
