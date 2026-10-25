import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.services.chunking import Chunk
from app.services.vector_service import VectorService

DIMS = 4
settings = get_settings().model_copy(
    update={"openai_embedding_dimensions": DIMS, "qdrant_collection": "unit_search"}
)
CAT = [1.0, 0.0, 0.0, 0.0]
DOG = [0.0, 1.0, 0.0, 0.0]
BOTH = [1.0, 1.0, 0.0, 0.0]


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def service(qdrant) -> VectorService:
    return VectorService(qdrant, settings)


async def store(service: VectorService, user, doc, items: list[tuple[str, list[float]]]) -> None:
    await service.upsert_chunks(
        user_id=user,
        document_id=doc,
        embedding_model="test",
        chunks=[Chunk(i, text) for i, (text, _) in enumerate(items)],
        vectors=[vector for _, vector in items],
    )


async def test_search_returns_best_matches_first_above_the_threshold(service):
    user, doc = uuid.uuid4(), uuid.uuid4()
    await store(service, user, doc, [("exact", CAT), ("partial", BOTH), ("unrelated", DOG)])

    results = await service.search(user_id=user, vector=CAT, limit=10, score_threshold=0.5)

    assert [r.text for r in results] == ["exact", "partial"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.707, abs=0.01)


async def test_search_never_returns_another_users_points(service):
    alice, bob = uuid.uuid4(), uuid.uuid4()
    alice_doc, bob_doc = uuid.uuid4(), uuid.uuid4()
    await store(service, alice, alice_doc, [("alice text", CAT)])
    await store(service, bob, bob_doc, [("bob text", CAT)])  # identical vector

    mine = await service.search(user_id=alice, vector=CAT, limit=10, score_threshold=0.0)
    wrong_owner = await service.search(
        user_id=alice, document_id=bob_doc, vector=CAT, limit=10, score_threshold=0.0
    )

    assert [r.text for r in mine] == ["alice text"]
    assert wrong_owner == []  # right document ID, wrong owner: nothing


async def test_search_can_be_limited_to_one_document(service):
    user, doc_a, doc_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await store(service, user, doc_a, [("from a", CAT)])
    await store(service, user, doc_b, [("from b", CAT)])

    results = await service.search(
        user_id=user, document_id=doc_b, vector=CAT, limit=10, score_threshold=0.0
    )

    assert [r.text for r in results] == ["from b"]


async def test_search_respects_the_limit(service):
    user, doc = uuid.uuid4(), uuid.uuid4()
    await store(service, user, doc, [(f"chunk {i}", CAT) for i in range(5)])

    results = await service.search(user_id=user, vector=CAT, limit=2, score_threshold=0.0)

    assert len(results) == 2


async def test_unknown_user_gets_nothing(service):
    await store(service, uuid.uuid4(), uuid.uuid4(), [("someone's text", CAT)])

    results = await service.search(user_id=uuid.uuid4(), vector=CAT, limit=10, score_threshold=0.0)

    assert results == []


async def test_result_fields(service):
    user, doc = uuid.uuid4(), uuid.uuid4()
    await service.upsert_chunks(
        user_id=user,
        document_id=doc,
        embedding_model="test",
        chunks=[Chunk(3, "hello")],
        vectors=[CAT],
    )

    (result,) = await service.search(user_id=user, vector=CAT, limit=5, score_threshold=0.0)

    assert result.document_id == doc
    assert result.chunk_index == 3
    assert result.text == "hello"
    assert isinstance(result.score, float)


class LeakyClient:
    """A misbehaving Qdrant that ignores filters and returns whatever it is given."""

    def __init__(self, points: list) -> None:
        self.points = points

    async def query_points(self, **kwargs):
        return SimpleNamespace(points=self.points)


def make_point(payload: dict, score: float = 0.9):
    return SimpleNamespace(payload=payload, score=score)


def leaky_service(points: list) -> VectorService:
    service = VectorService(LeakyClient(points), settings)
    service._ready = True  # skip collection setup
    return service


async def test_points_owned_by_someone_else_are_dropped_even_if_qdrant_returns_them():
    me = uuid.uuid4()
    mine = {"user_id": str(me), "document_id": str(uuid.uuid4()), "chunk_index": 0, "text": "ok"}
    theirs = {
        "user_id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "chunk_index": 0,
        "text": "leak",
    }
    service = leaky_service([make_point(theirs), make_point(mine)])

    results = await service.search(user_id=me, vector=CAT, limit=5, score_threshold=0.0)

    assert [r.text for r in results] == ["ok"]


async def test_malformed_points_are_skipped():
    me = uuid.uuid4()
    good = {"user_id": str(me), "document_id": str(uuid.uuid4()), "chunk_index": 1, "text": "ok"}
    bad_id = {"user_id": str(me), "document_id": "not-a-uuid", "chunk_index": 0, "text": "x"}
    service = leaky_service(
        [make_point(bad_id), make_point({"user_id": str(me)}), make_point(good)]
    )

    results = await service.search(user_id=me, vector=CAT, limit=5, score_threshold=0.0)

    assert [r.text for r in results] == ["ok"]


async def test_upstream_failures_become_external_service_errors():
    class Broken:
        async def query_points(self, **kwargs):
            raise ResponseHandlingException(Exception("down"))

    service = VectorService(Broken(), settings)
    service._ready = True

    with pytest.raises(ExternalServiceError):
        await service.search(user_id=uuid.uuid4(), vector=CAT, limit=5, score_threshold=0.0)
