"""Retrieval-augmented question answering."""

import logging
import time
import uuid
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ConflictError, NotFoundError
from app.models import DocumentStatus, User
from app.repositories.document_repository import DocumentRepository
from app.schemas.rag import AskRequest, AskResponse, SourceRead
from app.services.prompts import (
    NOT_FOUND_ANSWER,
    new_nonce,
    select_chunks,
    system_prompt,
    user_prompt,
)
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

OVERFETCH_FACTOR = 3  # candidates are fetched, then filtered by Postgres status
MAX_CANDIDATES = 100
SOURCE_EXCERPT_CHARS = 1500


class Embedder(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...


class Completer(Protocol):
    async def complete(self, *, system: str, user: str) -> str: ...


def _excerpt(text: str) -> str:
    if len(text) <= SOURCE_EXCERPT_CHARS:
        return text
    return text[: SOURCE_EXCERPT_CHARS - 1].rstrip() + "…"


def _is_refusal(answer: str) -> bool:
    normalized = answer.strip().rstrip(".").strip().lower()
    return normalized == NOT_FOUND_ANSWER.rstrip(".").lower()


class RagService:
    def __init__(
        self,
        db: AsyncSession,
        documents: DocumentRepository,
        vectors: VectorService,
        embeddings: Embedder,
        chat: Completer,
        settings: Settings,
    ) -> None:
        self.db = db
        self.documents = documents
        self.vectors = vectors
        self.embeddings = embeddings
        self.chat = chat
        self.settings = settings

    async def ask(self, user: User, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        user_id = user.id  # from the verified access token, never from the request body
        question = request.question
        top_k = self.settings.retrieval_top_k

        if request.document_id is not None:
            await self._require_searchable_document(user_id, request.document_id)
        await self._release_connection()  # no DB connection is held during slow calls below

        query_vector = await self.embeddings.embed_query(question)
        candidates = await self.vectors.search(
            user_id=user_id,
            document_id=request.document_id,
            vector=query_vector,
            limit=min(top_k * OVERFETCH_FACTOR, MAX_CANDIDATES),
            score_threshold=self.settings.retrieval_score_threshold,
        )

        # Qdrant doesn't know document status. Postgres decides what is searchable.
        names = await self._completed_document_names(user_id, {c.document_id for c in candidates})
        usable = [c for c in candidates if c.document_id in names][:top_k]
        selected = select_chunks(usable, self.settings.max_context_chars)

        if not selected:
            logger.info("Ask: no relevant context (%d candidates)", len(candidates))
            return self._not_found(question)

        nonce = new_nonce()
        answer = await self.chat.complete(
            system=system_prompt(nonce), user=user_prompt(question, selected, nonce)
        )
        if _is_refusal(answer):
            logger.info("Ask: the model found no answer in %d excerpt(s)", len(selected))
            return self._not_found(question)

        sources = [
            SourceRead(
                ref=ref,
                document_id=chunk.document_id,
                document_name=names[chunk.document_id],  # always current: read from Postgres
                chunk_index=chunk.chunk_index,
                content=_excerpt(chunk.text),
                score=chunk.score,
            )
            for ref, chunk in enumerate(selected, start=1)
        ]
        logger.info(
            "Ask answered: %d source(s), top score %.3f, %.0f ms",
            len(sources),
            selected[0].score,
            (time.perf_counter() - started) * 1000,
        )  # never the question, the context or the answer
        return AskResponse(question=question, answer=answer, context_found=True, sources=sources)

    # ---- internals --------------------------------------------------------------------

    async def _require_searchable_document(
        self, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> None:
        # The owner check lives inside the SQL WHERE, so foreign IDs look like missing ones.
        document = await self.documents.get_owned(document_id, user_id)
        if document is None:
            raise NotFoundError("Document not found.")
        if document.status != DocumentStatus.COMPLETED:
            raise ConflictError(
                f"This document is not ready to be searched (status: {document.status.value})."
            )

    async def _completed_document_names(
        self, user_id: uuid.UUID, document_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        if not document_ids:
            return {}
        names = await self.documents.get_completed_names(user_id, document_ids)
        await self._release_connection()
        return names

    async def _release_connection(self) -> None:
        # commit() ends the transaction and returns the connection to the pool. (The session
        # has expire_on_commit=False, so loaded objects such as `user` stay usable.)
        await self.db.commit()

    @staticmethod
    def _not_found(question: str) -> AskResponse:
        return AskResponse(
            question=question, answer=NOT_FOUND_ANSWER, context_found=False, sources=[]
        )
