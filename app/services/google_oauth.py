"""Google OAuth 2.0 / OpenID Connect: authorization-code flow with PKCE."""

import asyncio
import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from google.auth.exceptions import GoogleAuthError, TransportError
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import BadData, URLSafeTimedSerializer

from app.core.config import Settings
from app.core.exceptions import (
    BadRequestError,
    ExternalServiceError,
    ServiceUnavailableError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_EXCHANGE_URL = "https://oauth2.googleapis.com/token"
SIGN_IN_FAILED = "Google sign-in failed. Please try again."
UPSTREAM_DOWN = "Could not reach Google. Please try again."


@dataclass(frozen=True)
class GoogleIdentity:
    """Claims from a *verified* Google ID token."""

    subject: str
    email: str
    name: str | None
    picture: str | None


class GoogleOAuthService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._signer = URLSafeTimedSerializer(
            settings.session_secret_key.get_secret_value(), salt="google-oauth-flow"
        )

    # ---- step 1: start ----------------------------------------------------------------

    def begin_login(self) -> tuple[str, str]:
        """Return (google_authorization_url, signed_cookie_value)."""
        self._require_configured()
        state = secrets.token_urlsafe(32)  # CSRF protection for the redirect
        nonce = secrets.token_urlsafe(32)  # binds the ID token to this login attempt
        verifier = secrets.token_urlsafe(64)  # PKCE secret, 86 chars (spec: 43-128)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        params = {
            "client_id": self._settings.google_client_id,
            "redirect_uri": self._settings.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        url = f"{GOOGLE_AUTH_URL}?{urlencode(params, quote_via=quote)}"
        cookie = self._signer.dumps({"state": state, "nonce": nonce, "verifier": verifier})
        return url, cookie

    # ---- step 2: finish ---------------------------------------------------------------

    async def complete_login(
        self, *, code: str, state: str, cookie_value: str | None
    ) -> GoogleIdentity:
        self._require_configured()
        flow = self._load_flow(cookie_value)
        if not secrets.compare_digest(state.encode(), flow["state"].encode()):
            raise BadRequestError("Invalid sign-in state. Please start again.")

        token_payload = await self._exchange_code(code, flow["verifier"])
        raw_id_token = token_payload.get("id_token")
        if not isinstance(raw_id_token, str) or not raw_id_token:
            logger.warning("Google token response had no id_token")
            raise UnauthorizedError(SIGN_IN_FAILED)

        claims = await self._verify_id_token(raw_id_token)

        token_nonce = str(claims.get("nonce") or "")
        if not secrets.compare_digest(token_nonce.encode(), flow["nonce"].encode()):
            logger.warning("ID token nonce mismatch")
            raise UnauthorizedError(SIGN_IN_FAILED)

        email = claims.get("email")
        subject = claims.get("sub")
        if not email or not subject:
            raise UnauthorizedError(SIGN_IN_FAILED)
        if claims.get("email_verified") not in (True, "true"):
            raise UnauthorizedError("Your Google account's email address is not verified.")

        return GoogleIdentity(
            subject=str(subject),
            email=str(email),
            name=claims.get("name"),
            picture=claims.get("picture"),
        )

    # ---- internals --------------------------------------------------------------------

    def _require_configured(self) -> None:
        s = self._settings
        if not s.google_client_id or not s.google_client_secret.get_secret_value():
            raise ServiceUnavailableError("Google sign-in is not configured.")

    def _load_flow(self, cookie_value: str | None) -> dict[str, str]:
        expired = BadRequestError("Sign-in session expired or invalid. Please start again.")
        if not cookie_value:
            raise expired
        try:
            data = self._signer.loads(cookie_value, max_age=self._settings.oauth_state_ttl_seconds)
        except BadData:  # bad signature, tampering, or expired
            raise expired from None
        if not isinstance(data, dict) or not all(
            isinstance(data.get(k), str) for k in ("state", "nonce", "verifier")
        ):
            raise expired
        return data

    async def _exchange_code(self, code: str, verifier: str) -> dict[str, Any]:
        """Back-channel call: authorization code + PKCE verifier + client secret -> tokens."""
        s = self._settings
        form = {
            "code": code,
            "client_id": s.google_client_id,
            "client_secret": s.google_client_secret.get_secret_value(),
            "redirect_uri": s.google_redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(GOOGLE_EXCHANGE_URL, data=form)
        except httpx.HTTPError as exc:
            logger.warning("Google token endpoint unreachable: %s", type(exc).__name__)
            raise ExternalServiceError(UPSTREAM_DOWN) from None

        if response.status_code != 200:
            try:
                reason = str(response.json().get("error", ""))
            except ValueError:
                reason = ""
            # Log Google's error code only, never the request or response body.
            logger.warning("Google code exchange rejected: %s %s", response.status_code, reason)
            if response.status_code >= 500:
                raise ExternalServiceError(UPSTREAM_DOWN)
            raise UnauthorizedError(SIGN_IN_FAILED)
        try:
            return dict(response.json())
        except ValueError:
            raise ExternalServiceError(UPSTREAM_DOWN) from None

    async def _verify_id_token(self, raw_id_token: str) -> dict[str, Any]:
        """Verifies signature (Google's public keys), issuer, audience and expiry."""

        def _verify() -> dict[str, Any]:
            return google_id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                audience=self._settings.google_client_id,
                clock_skew_in_seconds=10,
            )

        try:
            # google-auth is synchronous (it may fetch Google's certs), so keep it off the loop.
            return await asyncio.wait_for(asyncio.to_thread(_verify), timeout=15)
        except (TransportError, TimeoutError):
            logger.warning("Could not fetch Google signing certificates")
            raise ExternalServiceError(UPSTREAM_DOWN) from None
        except (ValueError, GoogleAuthError) as exc:
            # Log the class only: library messages can include fragments of the token.
            logger.warning("Google ID token rejected: %s", type(exc).__name__)
            raise UnauthorizedError(SIGN_IN_FAILED) from None
