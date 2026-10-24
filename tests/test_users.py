from sqlalchemy import delete, update

from app.models import User

ME = "/api/v1/users/me"


def _bearer(response) -> dict[str, str]:
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def test_me_requires_authentication(api_client):
    resp = await api_client.get(ME)

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_me_rejects_malformed_token(api_client):
    resp = await api_client.get(ME, headers={"Authorization": "Bearer not.a.jwt"})

    assert resp.status_code == 401


async def test_me_returns_profile_without_internal_fields(sign_in, api_client):
    login = await sign_in()

    resp = await api_client.get(ME, headers=_bearer(login))

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert body["name"] == "Alice"
    assert set(body) == {"id", "email", "name", "avatar_url", "created_at"}


async def test_me_rejects_token_of_deleted_user(sign_in, api_client, session_factory):
    login = await sign_in()
    async with session_factory() as session:
        await session.execute(delete(User))
        await session.commit()

    resp = await api_client.get(ME, headers=_bearer(login))

    assert resp.status_code == 401


async def test_me_rejects_token_of_deactivated_user(sign_in, api_client, session_factory):
    login = await sign_in()
    async with session_factory() as session:
        await session.execute(update(User).values(is_active=False))
        await session.commit()

    resp = await api_client.get(ME, headers=_bearer(login))

    assert resp.status_code == 401
