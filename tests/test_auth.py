from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from sqlalchemy import func, select, update

from app.api.deps import get_google_oauth
from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.models import RefreshToken, User
from app.services.google_oauth import GoogleOAuthService

AUTH = "/api/v1/auth"


def _cookie(token: str) -> dict[str, str]:
    """Send a specific refresh token, regardless of what the client's cookie jar holds."""
    return {"Cookie": f"refresh_token={token}"}


# ---- login / callback ---------------------------------------------------------------


async def test_login_redirects_to_google_with_state_nonce_and_pkce(api_client):
    resp = await api_client.get(f"{AUTH}/google/login")

    assert resp.status_code == 302
    location = urlparse(resp.headers["location"])
    assert location.netloc == "accounts.google.com"
    query = parse_qs(location.query)
    assert query["response_type"] == ["code"]
    assert query["scope"] == ["openid email profile"]
    assert query["code_challenge_method"] == ["S256"]
    for key in ("client_id", "redirect_uri", "state", "nonce", "code_challenge"):
        assert query[key][0]
    cookie = resp.headers["set-cookie"].lower()
    assert cookie.startswith("oauth_state=")
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


async def test_login_returns_503_when_google_not_configured(api_app, api_client, test_settings):
    unconfigured = GoogleOAuthService(test_settings.model_copy(update={"google_client_id": ""}))
    api_app.dependency_overrides[get_google_oauth] = lambda: unconfigured

    resp = await api_client.get(f"{AUTH}/google/login")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_unavailable"


async def test_callback_creates_user_and_issues_tokens(sign_in, session_factory):
    resp = await sign_in()

    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] > 0
    assert resp.headers["cache-control"] == "no-store"

    cookies = [c.lower() for c in resp.headers.get_list("set-cookie")]
    refresh = next(c for c in cookies if c.startswith("refresh_token="))
    assert "httponly" in refresh
    assert "path=/api/v1/auth" in refresh
    assert "samesite=lax" in refresh
    cleared_state = next(c for c in cookies if c.startswith("oauth_state="))
    assert "max-age=0" in cleared_state

    async with session_factory() as session:
        user = (await session.execute(select(User))).scalar_one()
    assert user.email == "alice@example.com"
    assert user.google_subject_id == "google-sub-1"


async def test_callback_rejects_state_mismatch(api_client):
    await api_client.get(f"{AUTH}/google/login")

    resp = await api_client.get(
        f"{AUTH}/google/callback", params={"code": "x", "state": "attacker-chosen-state"}
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


async def test_callback_rejects_missing_state_cookie(api_client):
    resp = await api_client.get(f"{AUTH}/google/callback", params={"code": "x", "state": "abc"})

    assert resp.status_code == 400


async def test_callback_rejects_nonce_mismatch(sign_in):
    resp = await sign_in(nonce="forged-nonce")

    assert resp.status_code == 401


async def test_callback_rejects_unverified_email(sign_in, session_factory):
    resp = await sign_in(email_verified=False)

    assert resp.status_code == 401
    async with session_factory() as session:
        assert (await session.execute(select(func.count()).select_from(User))).scalar_one() == 0


async def test_callback_rejects_invalid_id_token(sign_in, google):
    google.verify_error = UnauthorizedError("Google sign-in failed. Please try again.")

    resp = await sign_in()

    assert resp.status_code == 401


async def test_callback_handles_user_denying_access(api_client):
    await api_client.get(f"{AUTH}/google/login")

    resp = await api_client.get(f"{AUTH}/google/callback", params={"error": "access_denied"})

    assert resp.status_code == 400
    assert "access_denied" not in resp.text


async def test_repeat_login_reuses_user_and_updates_profile(sign_in, session_factory):
    await sign_in()
    resp = await sign_in(name="Alice Updated")

    assert resp.status_code == 200
    async with session_factory() as session:
        users = (await session.execute(select(User))).scalars().all()
        families = (
            await session.execute(select(func.count(func.distinct(RefreshToken.family_id))))
        ).scalar_one()
    assert len(users) == 1
    assert users[0].name == "Alice Updated"
    assert families == 2  # each login starts its own session family


async def test_inactive_user_cannot_sign_in(sign_in, session_factory):
    await sign_in()
    async with session_factory() as session:
        await session.execute(update(User).values(is_active=False))
        await session.commit()

    resp = await sign_in()

    assert resp.status_code == 403


async def test_same_email_from_different_google_account_is_rejected(sign_in, session_factory):
    await sign_in()

    resp = await sign_in(sub="google-sub-2")

    assert resp.status_code == 409
    async with session_factory() as session:
        assert (await session.execute(select(func.count()).select_from(User))).scalar_one() == 1


# ---- refresh / logout ---------------------------------------------------------------


async def test_refresh_rotates_token_and_returns_working_access_token(sign_in, api_client):
    first = await sign_in()
    old = first.cookies["refresh_token"]

    resp = await api_client.post(f"{AUTH}/refresh", headers=_cookie(old))

    assert resp.status_code == 200
    assert resp.cookies["refresh_token"] != old
    bearer = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    me = await api_client.get("/api/v1/users/me", headers=bearer)
    assert me.status_code == 200


async def test_reused_refresh_token_revokes_entire_family(
    api_app, api_client, test_settings, sign_in
):
    strict = test_settings.model_copy(update={"refresh_reuse_grace_seconds": 0})
    api_app.dependency_overrides[get_settings] = lambda: strict
    first = await sign_in()
    token_1 = first.cookies["refresh_token"]
    rotated = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))
    token_2 = rotated.cookies["refresh_token"]

    replay = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))
    legit = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_2))

    assert replay.status_code == 401
    assert legit.status_code == 401  # the whole family was revoked


async def test_reuse_within_grace_window_does_not_revoke_family(sign_in, api_client):
    first = await sign_in()
    token_1 = first.cookies["refresh_token"]
    rotated = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))
    token_2 = rotated.cookies["refresh_token"]

    replay = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))
    legit = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_2))

    assert replay.status_code == 401
    assert legit.status_code == 200


async def test_refresh_without_cookie_is_401(api_client):
    resp = await api_client.post(f"{AUTH}/refresh")

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_refresh_with_unknown_token_is_401_and_clears_cookie(api_client):
    resp = await api_client.post(f"{AUTH}/refresh", headers=_cookie("not-a-real-token"))

    assert resp.status_code == 401
    cleared = resp.headers["set-cookie"].lower()
    assert cleared.startswith("refresh_token=")
    assert "max-age=0" in cleared


async def test_expired_refresh_token_is_rejected(sign_in, api_client, session_factory):
    first = await sign_in()
    token = first.cookies["refresh_token"]
    async with session_factory() as session:
        await session.execute(
            update(RefreshToken).values(expires_at=datetime.now(UTC) - timedelta(minutes=1))
        )
        await session.commit()

    resp = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token))

    assert resp.status_code == 401


async def test_refresh_rejects_foreign_origin(sign_in, api_client, test_settings):
    first = await sign_in()
    token = first.cookies["refresh_token"]

    blocked = await api_client.post(
        f"{AUTH}/refresh", headers={**_cookie(token), "Origin": "https://evil.example"}
    )
    allowed = await api_client.post(
        f"{AUTH}/refresh", headers={**_cookie(token), "Origin": test_settings.cors_origins[0]}
    )

    assert blocked.status_code == 403
    assert allowed.status_code == 200  # the blocked attempt did not consume the token


async def test_refresh_for_deactivated_user_is_rejected(sign_in, api_client, session_factory):
    first = await sign_in()
    token = first.cookies["refresh_token"]
    async with session_factory() as session:
        await session.execute(update(User).values(is_active=False))
        await session.commit()

    resp = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token))

    assert resp.status_code == 401
    async with session_factory() as session:
        revoked_at = (await session.execute(select(RefreshToken.revoked_at))).scalar_one()
    assert revoked_at is not None


async def test_logout_revokes_whole_session(sign_in, api_client):
    first = await sign_in()
    token_1 = first.cookies["refresh_token"]
    rotated = await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))
    token_2 = rotated.cookies["refresh_token"]

    out = await api_client.post(f"{AUTH}/logout", headers=_cookie(token_2))

    assert out.status_code == 204
    assert "max-age=0" in out.headers["set-cookie"].lower()
    assert (await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_2))).status_code == 401
    assert (await api_client.post(f"{AUTH}/refresh", headers=_cookie(token_1))).status_code == 401


async def test_logout_without_cookie_is_204(api_client):
    resp = await api_client.post(f"{AUTH}/logout")

    assert resp.status_code == 204
