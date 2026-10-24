import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_token,
)

settings = get_settings()
KEY = settings.jwt_secret_key.get_secret_value()


def _encode(key: str = KEY, algorithm: str | None = None, **overrides: object) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(uuid.uuid4()),
        "iss": settings.app_name,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "type": "access",
        **overrides,
    }
    return jwt.encode(payload, key, algorithm=algorithm or settings.jwt_algorithm)


def test_access_token_round_trip():
    user_id = uuid.uuid4()

    token, expires_in = create_access_token(user_id, settings)

    assert decode_access_token(token, settings) == user_id
    assert expires_in == settings.access_token_expire_minutes * 60


def test_expired_access_token_is_rejected():
    token = _encode(exp=datetime.now(UTC) - timedelta(minutes=5))

    with pytest.raises(UnauthorizedError):
        decode_access_token(token, settings)


def test_token_signed_with_another_key_is_rejected():
    with pytest.raises(UnauthorizedError):
        decode_access_token(_encode(key="x" * 64), settings)


def test_unsigned_alg_none_token_is_rejected():
    with pytest.raises(UnauthorizedError):
        decode_access_token(_encode(key="", algorithm="none"), settings)


def test_token_with_wrong_type_is_rejected():
    with pytest.raises(UnauthorizedError):
        decode_access_token(_encode(type="refresh"), settings)


def test_token_from_another_issuer_is_rejected():
    with pytest.raises(UnauthorizedError):
        decode_access_token(_encode(iss="someone-else"), settings)


def test_refresh_token_helpers():
    first, second = generate_refresh_token(), generate_refresh_token()

    assert first != second
    assert len(first) == 64
    assert hash_token(first) == hash_token(first)
    assert hash_token(first) != hash_token(second)
    assert len(hash_token(first)) == 64
    assert first not in hash_token(first)
