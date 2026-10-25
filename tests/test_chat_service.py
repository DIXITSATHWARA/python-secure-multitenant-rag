from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ServiceUnavailableError
from app.services.chat_service import ChatService

settings = get_settings()
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, status: int):
    return cls(
        "secret provider detail", response=httpx.Response(status, request=REQUEST), body=None
    )


def completion(content: str | None, finish: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(total_tokens=42),
    )


class FakeCompletions:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def make_service(completions: FakeCompletions, **updates) -> ChatService:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ChatService(client, settings.model_copy(update=updates))


async def test_returns_a_stripped_answer_and_sends_the_expected_request():
    completions = FakeCompletions(completion("  Hello [1]  "))

    answer = await make_service(completions).complete(system="SYS", user="USR")

    assert answer == "Hello [1]"
    call = completions.calls[0]
    assert call["model"] == settings.openai_chat_model
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert call["max_completion_tokens"] == settings.openai_chat_max_output_tokens
    assert call["temperature"] == settings.openai_chat_temperature


async def test_negative_temperature_is_not_sent():
    completions = FakeCompletions(completion("ok"))

    await make_service(completions, openai_chat_temperature=-1).complete(system="s", user="u")

    assert "temperature" not in completions.calls[0]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.InternalServerError, 500), ExternalServiceError),
        (status_error(openai.BadRequestError, 400), ExternalServiceError),
        (status_error(openai.AuthenticationError, 401), ExternalServiceError),
        (openai.APIConnectionError(request=REQUEST), ExternalServiceError),
        (openai.APITimeoutError(request=REQUEST), ExternalServiceError),
        (status_error(openai.RateLimitError, 429), ServiceUnavailableError),
    ],
)
async def test_provider_errors_become_our_errors_without_leaking_details(error, expected):
    service = make_service(FakeCompletions(error=error))

    with pytest.raises(expected) as caught:
        await service.complete(system="s", user="u")

    assert "secret" not in caught.value.message


async def test_rate_limit_asks_the_client_to_retry_later():
    service = make_service(FakeCompletions(error=status_error(openai.RateLimitError, 429)))

    with pytest.raises(ServiceUnavailableError) as caught:
        await service.complete(system="s", user="u")

    assert caught.value.headers == {"Retry-After": "10"}


async def test_empty_content_is_an_error():
    with pytest.raises(ExternalServiceError):
        await make_service(FakeCompletions(completion("   "))).complete(system="s", user="u")


async def test_no_choices_is_an_error():
    empty = SimpleNamespace(choices=[], usage=None)

    with pytest.raises(ExternalServiceError):
        await make_service(FakeCompletions(empty)).complete(system="s", user="u")
