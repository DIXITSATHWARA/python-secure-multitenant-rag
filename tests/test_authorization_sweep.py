import re
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import get_settings

PREFIX = "/api/v1"

# Every route that may be called without an access token. Anything else must answer 401.
# A new route that is not listed here, and forgets authentication, fails the sweep below.
PUBLIC = {
    ("GET", f"{PREFIX}/health"),
    ("GET", f"{PREFIX}/health/ready"),
    ("GET", f"{PREFIX}/auth/google/login"),
    ("GET", f"{PREFIX}/auth/google/callback"),
    ("POST", f"{PREFIX}/auth/refresh"),  # authenticated by the HttpOnly refresh cookie
    ("POST", f"{PREFIX}/auth/logout"),
}


def api_routes(app):
    """Yield (METHOD, path) for every API route, using FastAPI's public OpenAPI schema."""
    schema = app.openapi()
    for path, operations in schema["paths"].items():
        if not path.startswith(PREFIX):
            continue
        for method in operations:
            if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                yield method.upper(), path


def fill(path: str, value: str | None = None) -> str:
    return re.sub(r"\{[^}]+\}", value or str(uuid.uuid4()), path)


def test_public_allow_list_matches_real_routes(api_app):
    existing = set(api_routes(api_app))

    assert PUBLIC <= existing, f"stale allow-list entries: {PUBLIC - existing}"


async def test_every_route_requires_authentication_unless_declared_public(api_app, api_client):
    protected = []
    for method, path in api_routes(api_app):
        if (method, path) in PUBLIC:
            continue
        resp = await api_client.request(method, fill(path))
        assert resp.status_code == 401, f"{method} {path} is reachable without authentication"
        protected.append((method, path))

    assert len(protected) >= 7  # guards against a sweep that silently found nothing


async def test_every_document_route_hides_other_users_documents(
    api_app, api_client, upload, alice, bob
):
    doc_id = (await upload(alice)).json()["id"]
    swept = []
    for method, path in api_routes(api_app):
        if "{document_id}" not in path:
            continue
        resp = await api_client.request(method, fill(path, doc_id), headers=bob)
        assert resp.status_code == 404, f"{method} {path} leaks another user's document"
        swept.append((method, path))

    assert len(swept) >= 3
    mine = await api_client.get(f"{PREFIX}/documents/{doc_id}", headers=alice)
    assert mine.status_code == 200
    assert mine.json()["status"] == "pending"  # Bob's DELETE attempt changed nothing


def forged_token(kind: str, valid_sub: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": valid_sub,
        "iss": settings.app_name,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "type": "access",
    }
    key = settings.jwt_secret_key.get_secret_value()
    if kind == "wrong_key":
        return jwt.encode(payload, "x" * 64, algorithm="HS256")
    if kind == "alg_none":
        return jwt.encode(payload, "", algorithm="none")
    if kind == "expired":
        payload["exp"] = now - timedelta(minutes=5)
    elif kind == "wrong_issuer":
        payload["iss"] = "someone-else"
    elif kind == "refresh_type":
        payload["type"] = "refresh"
    elif kind == "unknown_user":
        payload["sub"] = str(uuid.uuid4())
    else:
        raise ValueError(kind)
    return jwt.encode(payload, key, algorithm=settings.jwt_algorithm)


@pytest.mark.parametrize(
    "kind",
    ["wrong_key", "alg_none", "expired", "wrong_issuer", "refresh_type", "unknown_user"],
)
async def test_forged_or_invalid_tokens_are_rejected(api_client, alice, kind):
    real = alice["Authorization"].split()[1]
    valid_sub = jwt.decode(real, options={"verify_signature": False})["sub"]
    headers = {"Authorization": f"Bearer {forged_token(kind, valid_sub)}"}

    resp = await api_client.get(f"{PREFIX}/users/me", headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("header", ["Basic YWJj", "Bearer", "Bearer ", "token abc"])
async def test_malformed_authorization_headers_are_rejected(api_client, header):
    resp = await api_client.get(f"{PREFIX}/users/me", headers={"Authorization": header})

    assert resp.status_code == 401
