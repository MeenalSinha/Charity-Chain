"""
test_jwt_tokens.py - Unit tests for JWT creation, claims validation & expiration
"""

import pytest
import jwt
from datetime import timedelta
from backend.app.core.config import settings
from backend.app.core.constants import PlatformRole
from backend.app.core.security import (
    create_access_token, decode_access_token,
    TokenExpiredException, InvalidTokenException
)


def test_create_and_decode_jwt():
    """Verify standard JWT claims, subject, and role preservation."""
    user_id = "user-uuid-1234"
    role = PlatformRole.DONOR.value
    session_id = "sess-uuid-5678"

    token = create_access_token(
        subject=user_id,
        platform_role=role,
        session_id=session_id
    )
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["role"] == role
    assert payload["session_id"] == session_id
    assert payload["iss"] == "charitychain"
    assert payload["aud"] == "charitychain-api"
    assert "jti" in payload
    assert "exp" in payload


def test_expired_token_raises_exception():
    """Verify expired token raises TokenExpiredException."""
    token = create_access_token(
        subject="user-uuid-999",
        platform_role="DONOR",
        expires_delta=timedelta(seconds=-10)  # Expired in past
    )

    with pytest.raises(TokenExpiredException):
        decode_access_token(token)


def test_tampered_token_raises_exception():
    """Verify token signature tampering is detected."""
    token = create_access_token(subject="user-uuid-888", platform_role="DONOR")
    tampered = token[:-4] + "abcd"

    with pytest.raises(InvalidTokenException):
        decode_access_token(tampered)


def test_empty_token_raises_exception():
    """Verify empty or None token is rejected."""
    with pytest.raises(InvalidTokenException):
        decode_access_token("")

