import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import NotFoundError
from app.main import create_app


@pytest.fixture
def app() -> FastAPI:
    application = create_app()

    @application.get("/_test/not-found")
    async def _not_found() -> None:
        raise NotFoundError("Thing not found")

    @application.get("/_test/boom")
    async def _boom() -> None:
        raise RuntimeError("secret internal detail")

    @application.get("/_test/validate")
    async def _validate(limit: int) -> dict[str, int]:
        return {"limit": limit}

    return application


@pytest.fixture
async def client(app: FastAPI):
    # raise_app_exceptions=False lets us observe the 500 response instead of the exception.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_liveness_ok_and_request_id_header(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers["x-request-id"]


async def test_unknown_route_uses_standard_error_shape(client: AsyncClient) -> None:
    resp = await client.get("/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_app_error_is_formatted(client: AsyncClient) -> None:
    resp = await client.get("/_test/not-found")
    body = resp.json()["error"]
    assert resp.status_code == 404
    assert body["code"] == "not_found"
    assert body["message"] == "Thing not found"
    assert body["request_id"] == resp.headers["x-request-id"]


async def test_validation_error_does_not_echo_input(client: AsyncClient) -> None:
    resp = await client.get("/_test/validate", params={"limit": "abc"})
    body = resp.json()["error"]
    assert resp.status_code == 422
    assert body["code"] == "validation_error"
    assert body["details"][0]["field"] == "query.limit"
    assert "abc" not in resp.text


async def test_unhandled_error_is_masked(client: AsyncClient) -> None:
    resp = await client.get("/_test/boom")
    body = resp.json()["error"]
    assert resp.status_code == 500
    assert body["code"] == "internal_error"
    assert "secret internal detail" not in resp.text
    assert body["request_id"] == resp.headers["x-request-id"]
