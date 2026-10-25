import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.services.chunking import Chunk
from app.services.vector_service import VectorService, chunk_point_id

DIMS = 4
settings = get_settings().model_copy(
    update={"openai_embedding_dimensions": DIMS, "qdrant_collection": "unit_chunks"}
)


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
def service(qdrant) -> VectorService:
    return VectorService(qdrant, settings)


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(i, f"chunk {i}") for i in range(n)]


def _vectors(n: int) -> list[list[float]]:
    return [[1.0, float(i + 1), 0.5, 0.25] for i in range(n)]


async def _store(service: VectorService, user_id, doc_id, n: int = 3) -> None:
    await service.upsert_chunks(
        user_id=user_id,
        document_id=doc_id,
        embedding_model="test-model",
        chunks=_chunks(n),
        vectors=_vectors(n),
    )


async def test_collection_is_created_once_and_never_recreated(qdrant, service):
    user, doc = uuid.uuid4(), uuid.uuid4()
    await service.ensure_collection()
    assert (await qdrant.get_collection("unit_chunks")).config.params.vectors.size == DIMS
    await _store(service, user, doc)

    fresh = VectorService(qdrant, settings)  # e.g. the worker starting up later
    await fresh.ensure_collection()

    assert await fresh.count_document_points(user_id=user, document_id=doc) == 3


async def test_dimension_mismatch_is_a_loud_configuration_error(qdrant, service):
    await service.ensure_collection()
    other = VectorService(qdrant, settings.model_copy(update={"openai_embedding_dimensions": 8}))

    with pytest.raises(RuntimeError, match="dimensional"):
        await other.ensure_collection()


async def test_upserting_twice_does_not_duplicate_points(qdrant, service):
    user, doc = uuid.uuid4(), uuid.uuid4()

    await _store(service, user, doc)
    await _store(service, user, doc)

    assert await service.count_document_points(user_id=user, document_id=doc) == 3
    (record,) = await qdrant.retrieve(
        "unit_chunks", ids=[chunk_point_id(doc, 1)], with_payload=True
    )
    assert record.payload == {
        "user_id": str(user),
        "document_id": str(doc),
        "chunk_index": 1,
        "text": "chunk 1",
        "embedding_model": "test-model",
    }


async def test_delete_is_scoped_to_owner_and_document(service):
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    await _store(service, user_a, doc_a)
    await _store(service, user_b, doc_b)

    await service.delete_document(user_id=user_b, document_id=doc_a)  # wrong owner: no effect
    assert await service.count_document_points(user_id=user_a, document_id=doc_a) == 3

    await service.delete_document(user_id=user_a, document_id=doc_a)
    assert await service.count_document_points(user_id=user_a, document_id=doc_a) == 0
    assert await service.count_document_points(user_id=user_b, document_id=doc_b) == 3


async def test_deleting_nonexistent_document_is_not_an_error(service):
    await service.delete_document(user_id=uuid.uuid4(), document_id=uuid.uuid4())


def test_point_ids_are_deterministic_and_unique():
    doc, other = uuid.uuid4(), uuid.uuid4()

    assert chunk_point_id(doc, 0) == chunk_point_id(doc, 0)
    ids = {chunk_point_id(d, i) for d in (doc, other) for i in range(100)}
    assert len(ids) == 200
    uuid.UUID(chunk_point_id(doc, 0))  # valid UUID string


async def test_mismatched_chunks_and_vectors_are_rejected(service):
    with pytest.raises(ValueError):
        await service.upsert_chunks(
            user_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            embedding_model="m",
            chunks=_chunks(2),
            vectors=_vectors(1),
        )


async def test_ping_is_true_for_a_working_client(service):
    assert await service.ping() is True


async def test_ping_is_false_when_the_client_fails():
    class Broken:
        async def get_collections(self):
            raise ResponseHandlingException(Exception("down"))

    assert await VectorService(Broken(), settings).ping() is False


async def test_upstream_failures_become_external_service_errors():
    class Broken:
        async def collection_exists(self, *args, **kwargs):
            raise ResponseHandlingException(Exception("down"))

    with pytest.raises(ExternalServiceError):
        await VectorService(Broken(), settings).ensure_collection()
