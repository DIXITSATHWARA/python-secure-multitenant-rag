import pytest
from sqlalchemy import func, select

from app.api.deps import get_chat_service, get_embedding_service, get_vector_service
from app.core.config import get_settings
from app.models import Document
from tests.helpers import files_on_disk

ASK = "/api/v1/ask"


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.0]


class EmptyVectors:
    """No results, so /ask answers "not found" without calling the chat model."""

    async def search(self, **kwargs):
        return []


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture(autouse=True)
def wire(api_app, embeddings) -> None:
    api_app.dependency_overrides[get_vector_service] = lambda: EmptyVectors()
    api_app.dependency_overrides[get_embedding_service] = lambda: embeddings
    api_app.dependency_overrides[get_chat_service] = lambda: object()


def tighten(api_app, test_settings, **updates) -> None:
    api_app.dependency_overrides[get_settings] = lambda: test_settings.model_copy(update=updates)


async def ask(client, headers):
    return await client.post(ASK, json={"question": "hello"}, headers=headers)


async def test_ask_is_limited_per_user_before_any_paid_call_is_made(
    api_app, api_client, test_settings, alice, embeddings
):
    tighten(api_app, test_settings, rate_limit_ask_per_minute=2)

    responses = [await ask(api_client, alice) for _ in range(3)]

    assert [r.status_code for r in responses] == [200, 200, 429]
    blocked = responses[2]
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert blocked.json()["error"]["request_id"] == blocked.headers["x-request-id"]
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert embeddings.calls == 2  # the blocked request never reached OpenAI


async def test_one_users_limit_does_not_affect_another_user(
    api_app, api_client, test_settings, alice, bob
):
    tighten(api_app, test_settings, rate_limit_ask_per_minute=1)
    await ask(api_client, alice)

    assert (await ask(api_client, alice)).status_code == 429
    assert (await ask(api_client, bob)).status_code == 200


async def test_rate_limiting_can_be_switched_off(api_app, api_client, test_settings, alice):
    tighten(api_app, test_settings, rate_limit_enabled=False, rate_limit_ask_per_minute=1)

    responses = [await ask(api_client, alice) for _ in range(4)]

    assert {r.status_code for r in responses} == {200}


async def test_uploads_are_limited_per_user_before_any_work_is_done(
    api_app, test_settings, upload, alice, session_factory, storage
):
    tighten(api_app, test_settings, rate_limit_upload_per_minute=2)

    responses = [await upload(alice, content=f"doc {i}".encode()) for i in range(3)]

    assert [r.status_code for r in responses] == [202, 202, 429]
    assert responses[2].headers["retry-after"]
    assert len(files_on_disk(storage)) == 2  # the refused upload stored nothing
    async with session_factory() as db:
        assert (await db.execute(select(func.count()).select_from(Document))).scalar_one() == 2


async def test_auth_endpoints_are_limited_per_ip(api_client):
    settings = get_settings()
    limit = settings.rate_limit_auth_per_minute
    if not settings.rate_limit_enabled or limit <= 0:
        pytest.skip("the per-IP auth limit is disabled in .env")

    for _ in range(limit):
        assert (await api_client.post("/api/v1/auth/refresh")).status_code == 401
    blocked = await api_client.post("/api/v1/auth/refresh")

    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert (await api_client.get("/api/v1/health")).status_code == 200  # other paths unaffected


async def test_upload_endpoint_has_a_per_ip_guard_before_authentication(api_client):
    settings = get_settings()
    limit = settings.rate_limit_upload_ip_per_minute
    if not settings.rate_limit_enabled or limit <= 0:
        pytest.skip("the per-IP upload limit is disabled in .env")

    for _ in range(limit):
        assert (await api_client.post("/api/v1/documents")).status_code == 401
    blocked = await api_client.post("/api/v1/documents")

    assert blocked.status_code == 429


async def test_document_quota_is_enforced_per_user(
    api_app, test_settings, upload, alice, bob, storage
):
    tighten(api_app, test_settings, max_documents_per_user=2)

    first, second, third = [await upload(alice, content=f"doc {i}".encode()) for i in range(3)]

    assert [first.status_code, second.status_code, third.status_code] == [202, 202, 403]
    assert third.json()["error"]["code"] == "quota_exceeded"
    assert len(files_on_disk(storage)) == 2
    assert (await upload(bob)).status_code == 202  # Bob has his own quota


async def test_an_idempotent_replay_is_allowed_at_the_quota(api_app, test_settings, upload, alice):
    tighten(api_app, test_settings, max_documents_per_user=1)

    first = await upload(alice, key="quota-key-001")
    replay = await upload(alice, key="quota-key-001")
    another = await upload(alice, content=b"a different file")

    assert (first.status_code, replay.status_code, another.status_code) == (202, 200, 403)
