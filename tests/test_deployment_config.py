from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app


def build_app(monkeypatch, **updates):
    settings = get_settings().model_copy(update=updates)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    return create_app()


def client(app, host: str) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url=f"http://{host}")


async def test_unknown_host_headers_are_rejected(monkeypatch):
    app = build_app(monkeypatch, allowed_hosts=["api.example.com"])

    async with client(app, "evil.example") as bad, client(app, "api.example.com") as good:
        assert (await bad.get("/api/v1/health")).status_code == 400
        assert (await good.get("/api/v1/health")).status_code == 200


async def test_without_an_allow_list_any_host_is_accepted(monkeypatch):
    app = build_app(monkeypatch, allowed_hosts=[])

    async with client(app, "anything.example") as c:
        assert (await c.get("/api/v1/health")).status_code == 200


async def test_production_hides_docs_and_sends_hsts(monkeypatch):
    app = build_app(
        monkeypatch, app_env="production", cookie_secure=True, allowed_hosts=["api.example.com"]
    )

    async with client(app, "api.example.com") as c:
        docs = await c.get("/docs")
        health = await c.get("/api/v1/health")

    assert docs.status_code == 404
    assert health.status_code == 200
    assert "max-age=" in health.headers["strict-transport-security"]
