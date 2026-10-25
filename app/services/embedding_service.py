import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.services.errors import PermanentProcessingError

logger = logging.getLogger(__name__)

UNAVAILABLE = "The embedding service is unavailable. Please try again."


class EmbeddingService:
    """Wraps OpenAI embeddings. The same instance type embeds documents and, in Phase 6,
    questions, so both always use the same model and dimensions."""

    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._model = settings.openai_embedding_model
        self._dimensions = settings.openai_embedding_dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except BadRequestError as exc:
            logger.warning("OpenAI rejected the embedding input (HTTP %s)", exc.status_code)
            raise PermanentProcessingError("The document text could not be embedded.") from None
        except RateLimitError:
            logger.warning("OpenAI rate limit reached")
            raise ExternalServiceError(UNAVAILABLE) from None
        except (APIConnectionError, APIStatusError, OpenAIError) as exc:
            # Class name and status only. Never log request bodies (they hold document text).
            logger.warning(
                "OpenAI embeddings call failed: %s (status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise ExternalServiceError(UNAVAILABLE) from None

        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        if len(vectors) != len(texts) or any(len(v) != self._dimensions for v in vectors):
            logger.error("OpenAI returned an unexpected number or size of embeddings")
            raise ExternalServiceError("The embedding service returned an unexpected response.")
        if response.usage:
            logger.info("Embedded %d texts (%d tokens)", len(texts), response.usage.total_tokens)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]
