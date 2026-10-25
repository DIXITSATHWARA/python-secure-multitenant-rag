from app.api.deps import get_vector_service


class FakeVectors:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    async def ping(self) -> bool:
        return self.healthy


async def test_ready_when_all_dependencies_are_up(api_app, api_client):
    api_app.dependency_overrides[get_vector_service] = lambda: FakeVectors(True)

    resp = await api_client.get("/api/v1/health/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {"postgres": "ok", "qdrant": "ok"}}


async def test_degraded_when_qdrant_is_down(api_app, api_client):
    api_app.dependency_overrides[get_vector_service] = lambda: FakeVectors(False)

    resp = await api_client.get("/api/v1/health/ready")

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "degraded",
        "checks": {"postgres": "ok", "qdrant": "unavailable"},
    }


async def test_degraded_when_vector_service_was_never_created(api_app, api_client):
    api_app.dependency_overrides[get_vector_service] = lambda: None

    resp = await api_client.get("/api/v1/health/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["qdrant"] == "unavailable"
