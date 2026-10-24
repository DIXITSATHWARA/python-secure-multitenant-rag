from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

import app.models  # noqa: F401  (registers every model on Base.metadata)
from app.api.deps import get_google_oauth, get_storage
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.services.google_oauth import GoogleOAuthService
from app.services.storage import LocalStorage

_schema_ready = False


@pytest.fixture
def test_settings() -> Settings:
    return get_settings().model_copy(
        update={
            "google_client_id": "test-client-id.apps.googleusercontent.com",
            "google_client_secret": SecretStr("test-client-secret"),
            "cookie_secure": False,
            "frontend_redirect_url": "",
            "refresh_reuse_grace_seconds": 10,
        }
    )


@pytest.fixture
def test_db_url() -> str:
    settings = get_settings()
    url = settings.test_database_url
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; skipping database tests")
    if url == settings.database_url or "test" not in (make_url(url).database or ""):
        pytest.fail("TEST_DATABASE_URL must be a separate database whose name contains 'test'")
    return url


@pytest.fixture
async def db_engine(test_db_url: str) -> AsyncIterator[AsyncEngine]:
    """Fresh schema once per run, empty tables before every test."""
    global _schema_ready
    engine = create_async_engine(test_db_url, poolclass=NullPool)
    async with engine.begin() as conn:
        if not _schema_ready:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            _schema_ready = True
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Each test gets its own throwaway upload directory."""
    return LocalStorage(tmp_path / "storage")


class GoogleHarness:
    """The real GoogleOAuthService, with its two network calls replaced by fakes."""

    def __init__(self, settings: Settings) -> None:
        self.service = GoogleOAuthService(settings)
        self.nonce: str | None = None
        self.claims: dict[str, Any] = {}  # overrides for the fake ID-token claims
        self.verify_error: Exception | None = None
        self.service._exchange_code = self._exchange_code
        self.service._verify_id_token = self._verify_id_token

    async def _exchange_code(self, code: str, verifier: str) -> dict[str, Any]:
        return {"id_token": "fake-id-token"}

    async def _verify_id_token(self, raw_id_token: str) -> dict[str, Any]:
        if self.verify_error is not None:
            raise self.verify_error
        return {
            "sub": "google-sub-1",
            "email": "Alice@Example.com",
            "email_verified": True,
            "name": "Alice",
            "picture": "https://example.com/alice.png",
            "nonce": self.nonce,
            **self.claims,
        }


@pytest.fixture
def google(test_settings: Settings) -> GoogleHarness:
    return GoogleHarness(test_settings)


@pytest.fixture
def api_app(
    db_engine: AsyncEngine,
    test_settings: Settings,
    google: GoogleHarness,
    storage: LocalStorage,
) -> FastAPI:
    application = create_app()
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_settings] = lambda: test_settings
    application.dependency_overrides[get_google_oauth] = lambda: google.service
    application.dependency_overrides[get_storage] = lambda: storage
    return application


@pytest.fixture
async def api_client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=api_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sign_in(api_client: AsyncClient, google: GoogleHarness) -> Callable[..., Awaitable[Response]]:
    """Runs login + callback. Keyword args override the fake Google ID-token claims."""

    async def _sign_in(**claims: Any) -> Response:
        google.claims = claims
        login = await api_client.get("/api/v1/auth/google/login")
        params = parse_qs(urlparse(login.headers["location"]).query)
        google.nonce = params["nonce"][0]
        return await api_client.get(
            "/api/v1/auth/google/callback",
            params={"code": "auth-code", "state": params["state"][0]},
        )

    return _sign_in


async def _auth_headers(
    sign_in: Callable[..., Awaitable[Response]], **claims: Any
) -> dict[str, str]:
    response = await sign_in(**claims)
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def alice(sign_in: Callable[..., Awaitable[Response]]) -> dict[str, str]:
    """Authorization headers for user Alice."""
    return await _auth_headers(
        sign_in, sub="google-sub-alice", email="alice@example.com", name="Alice"
    )


@pytest.fixture
async def bob(sign_in: Callable[..., Awaitable[Response]]) -> dict[str, str]:
    """Authorization headers for user Bob."""
    return await _auth_headers(sign_in, sub="google-sub-bob", email="bob@example.com", name="Bob")


@pytest.fixture
def upload(api_client: AsyncClient) -> Callable[..., Awaitable[Response]]:
    async def _upload(
        headers: dict[str, str],
        *,
        filename: str = "notes.txt",
        content: bytes = b"hello world",
        name: str | None = None,
        key: str | None = None,
    ) -> Response:
        request_headers = dict(headers)
        if key:
            request_headers["Idempotency-Key"] = key
        return await api_client.post(
            "/api/v1/documents",
            files={"file": (filename, content, "application/octet-stream")},
            data={"name": name} if name else {},
            headers=request_headers,
        )

    return _upload
