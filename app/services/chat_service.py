import logging
from typing import Any

from openai import APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class ChatService:
    """Wraps the OpenAI chat call and translates every failure into our own error types.
    Provider messages are logged by class and status only: they can echo prompt content."""

    def __init__(self, client: AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._model = settings.openai_chat_model
        self._max_tokens = settings.openai_chat_max_output_tokens
        self._temperature = settings.openai_chat_temperature

    async def complete(self, *, system: str, user: str) -> str:
        options: dict[str, Any] = {}
        if self._temperature >= 0:
            options["temperature"] = self._temperature
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self._max_tokens,
                **options,
            )
        except RateLimitError as exc:
            # Also raised when billing quota is exhausted. Check the status in the log.
            logger.warning("OpenAI chat rate limit (status=%s)", exc.status_code)
            raise ServiceUnavailableError(
                "The AI service is busy. Please try again in a few seconds.",
                headers={"Retry-After": "10"},
            ) from None
        except APITimeoutError:
            logger.warning("OpenAI chat call timed out")
            raise ExternalServiceError("The AI service timed out. Please try again.") from None
        except OpenAIError as exc:
            logger.warning(
                "OpenAI chat call failed: %s (status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            raise ExternalServiceError("The AI service failed. Please try again.") from None

        choice = response.choices[0] if response.choices else None
        content = ((choice.message.content or "") if choice else "").strip()
        if not content:
            logger.warning("OpenAI chat call returned no content")
            raise ExternalServiceError("The AI service returned an empty answer. Please retry.")
        if choice.finish_reason == "length":
            logger.warning("Answer was cut off by OPENAI_CHAT_MAX_OUTPUT_TOKENS")
        if response.usage:
            logger.info("Chat completion used %d tokens", response.usage.total_tokens)
        return content
