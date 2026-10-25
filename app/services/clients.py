from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    # The SDK retries 429/5xx/timeouts with backoff and honors Retry-After.
    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def build_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=settings.qdrant_timeout_seconds,
    )
