"""All Qdrant access lives here, so tenant-isolation filters have exactly one home."""

import asyncio
import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)


def chunk_point_id(document_id: uuid.UUID, index: int) -> str:
    """Deterministic point ID: re-processing a document overwrites its points, never duplicates."""
    return str(uuid.uuid5(document_id, str(index)))


@contextmanager
def _upstream(action: str) -> Iterator[None]:
    try:
        yield
    except (ApiException, ResponseHandlingException, OSError, TimeoutError) as exc:
        logger.warning("Qdrant %s failed: %s", action, type(exc).__name__)
        raise ExternalServiceError(
            "The vector database is unavailable. Please try again."
        ) from None


class VectorService:
    def __init__(self, client: AsyncQdrantClient, settings: Settings) -> None:
        self._client = client
        self._collection = settings.qdrant_collection
        self._dimensions = settings.openai_embedding_dimensions
        self._ready = False
        self._lock = asyncio.Lock()

    # ---- collection setup -------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the collection and indexes only if missing. Safe to call repeatedly."""
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            with _upstream("ensure_collection"):
                await self._ensure_collection()
            self._ready = True

    async def _ensure_collection(self) -> None:
        client = self._client
        if not await client.collection_exists(self._collection):
            try:
                await client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimensions, distance=models.Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection %s", self._collection)
            except (ApiException, ResponseHandlingException):
                # Another process may have created it a moment ago.
                if not await client.collection_exists(self._collection):
                    raise

        info = await client.get_collection(self._collection)
        self._check_dimensions(info)

        indexed = set((getattr(info, "payload_schema", None) or {}).keys())
        if "user_id" not in indexed:
            await client.create_payload_index(
                collection_name=self._collection,
                field_name="user_id",
                # is_tenant: Qdrant co-locates each user's vectors for fast filtered search.
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )
        if "document_id" not in indexed:
            await client.create_payload_index(
                collection_name=self._collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _check_dimensions(self, info: models.CollectionInfo) -> None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise RuntimeError(f"Collection '{self._collection}' uses named vectors (unsupported).")
        if vectors.size != self._dimensions:
            raise RuntimeError(
                f"Qdrant collection '{self._collection}' has {vectors.size}-dimensional vectors "
                f"but OPENAI_EMBEDDING_DIMENSIONS={self._dimensions}. Use a different "
                "QDRANT_COLLECTION, or recreate the collection and re-embed your documents."
            )

    # ---- writes -----------------------------------------------------------------------

    async def upsert_chunks(
        self,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        embedding_model: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        await self.ensure_collection()
        points = [
            models.PointStruct(
                id=chunk_point_id(document_id, chunk.index),
                vector=list(vector),
                payload={
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "embedding_model": embedding_model,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with _upstream("upsert"):
            # wait=True: don't return until the points are durable.
            await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def delete_document(self, *, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Idempotent. Requires BOTH ids to match, so a wrong owner deletes nothing."""
        await self.ensure_collection()
        with _upstream("delete"):
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=self.owner_filter(user_id, document_id)
                ),
                wait=True,
            )

    # ---- reads ------------------------------------------------------------------------

    async def count_document_points(self, *, user_id: uuid.UUID, document_id: uuid.UUID) -> int:
        await self.ensure_collection()
        with _upstream("count"):
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=self.owner_filter(user_id, document_id),
                exact=True,
            )
        return result.count

    async def ping(self) -> bool:
        try:
            await asyncio.wait_for(self._client.get_collections(), timeout=3)
        except Exception:
            logger.warning("Qdrant readiness check failed", exc_info=True)
            return False
        return True

    @staticmethod
    def owner_filter(user_id: uuid.UUID, document_id: uuid.UUID | None = None) -> models.Filter:
        """The mandatory tenant filter. Phase 6 search uses this too. user_id always comes
        from the authenticated user, never from the request."""
        conditions: list[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
        ]
        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=str(document_id))
                )
            )
        return models.Filter(must=conditions)
