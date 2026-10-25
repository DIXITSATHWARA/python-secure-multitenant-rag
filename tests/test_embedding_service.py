from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.services.embedding_service import EmbeddingService
from app.services.errors import PermanentProcessingError

settings = get_settings().model_copy(update={"openai_embedding_dimensions": 3})
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def status_error(cls, status: int):
    response = httpx.Response(status, request=REQUEST)
    return cls("secret provider detail", response=response, body=None)


def item(index: int, vector: list[float]):
    return SimpleNamespace(index=index, embedding=vector)


def reply(items, tokens: int | None = 7):
    usage = SimpleNamespace(total_tokens=tokens) if tokens is not None else None
    return SimpleNamespace(data=items, usage=usage)


class FakeEmbeddingsApi:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def make_service(api: FakeEmbeddingsApi) -> EmbeddingService:
    return EmbeddingService(SimpleNamespace(embeddings=api), settings)


async def test_sends_the_configured_model_and_returns_vectors_in_input_order():
    api = FakeEmbeddingsApi(reply([item(1, [4.0, 5.0, 6.0]), item(0, [1.0, 2.0, 3.0])]))

    vectors = await make_service(api).embed_texts(["first", "second"])

    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    call = api.calls[0]
    assert call["model"] == settings.openai_embedding_model
    assert call["input"] == ["first", "second"]
    assert call["dimensions"] == 3
    assert call["encoding_format"] == "float"


async def test_empty_input_makes_no_api_call():
    api = FakeEmbeddingsApi(reply([]))

    assert await make_service(api).embed_texts([]) == []
    assert api.calls == []


async def test_embed_query_returns_a_single_vector():
    api = FakeEmbeddingsApi(reply([item(0, [0.1, 0.2, 0.3])]))

    assert await make_service(api).embed_query("hello") == [0.1, 0.2, 0.3]


async def test_input_rejected_by_openai_is_a_permanent_error():
    api = FakeEmbeddingsApi(error=status_error(openai.BadRequestError, 400))

    with pytest.raises(PermanentProcessingError) as caught:
        await make_service(api).embed_texts(["x"])

    assert "secret" not in caught.value.message


async def test_rate_limit_is_a_retryable_external_error():
    api = FakeEmbeddingsApi(error=status_error(openai.RateLimitError, 429))

    with pytest.raises(ExternalServiceError) as caught:
        await make_service(api).embed_texts(["x"])

    assert "secret" not in caught.value.message


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.InternalServerError, 500),
        status_error(openai.AuthenticationError, 401),
        openai.APIConnectionError(request=REQUEST),
        openai.APITimeoutError(request=REQUEST),
    ],
)
async def test_other_provider_failures_are_retryable_and_leak_nothing(error):
    api = FakeEmbeddingsApi(error=error)

    with pytest.raises(ExternalServiceError) as caught:
        await make_service(api).embed_texts(["x"])

    assert "secret" not in caught.value.message


async def test_wrong_number_of_vectors_is_an_error():
    api = FakeEmbeddingsApi(reply([item(0, [1.0, 2.0, 3.0])]))

    with pytest.raises(ExternalServiceError):
        await make_service(api).embed_texts(["a", "b"])


async def test_wrong_vector_size_is_an_error():
    api = FakeEmbeddingsApi(reply([item(0, [1.0, 2.0])]))

    with pytest.raises(ExternalServiceError):
        await make_service(api).embed_texts(["a"])


async def test_missing_usage_information_is_tolerated():
    api = FakeEmbeddingsApi(reply([item(0, [1.0, 2.0, 3.0])], tokens=None))

    assert await make_service(api).embed_texts(["a"]) == [[1.0, 2.0, 3.0]]
