import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import Settings
from app.core.exceptions import UnauthorizedError

ACCESS_JWT_KIND = "access"
INVALID_ACCESS = "Invalid or expired access token."


def create_access_token(user_id: uuid.UUID, settings: Settings) -> tuple[str, int]:
    """Return (jwt, lifetime in seconds)."""
    now = datetime.now(UTC)
    lifetime = timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": str(user_id),
        "iss": settings.app_name,
        "iat": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
        "type": ACCESS_JWT_KIND,
    }
    encoded = jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return encoded, int(lifetime.total_seconds())


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    """Validate signature, expiry, issuer and type. Returns the user ID."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            # Pin the algorithm list. Never trust the token's own "alg" header.
            algorithms=[settings.jwt_algorithm],
            issuer=settings.app_name,
            leeway=5,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
        if payload.get("type") != ACCESS_JWT_KIND:
            raise UnauthorizedError(INVALID_ACCESS)
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError, TypeError):
        raise UnauthorizedError(INVALID_ACCESS) from None


def generate_refresh_token() -> str:
    """64 URL-safe characters of cryptographic randomness."""
    return secrets.token_urlsafe(48)


def hash_token(raw: str) -> str:
    """SHA-256 is right for high-entropy random tokens (slow password hashes aren't needed)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
