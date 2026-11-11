"""
test_password_security.py - Unit tests for Argon2id hashing & token security
"""

import pytest
from backend.app.core.security import (
    hash_password, verify_password, validate_password_strength,
    generate_secure_token, hash_token, verify_token_hash, normalize_email
)


def test_argon2_hash_and_verify():
    """Verify Argon2id hashes passwords securely and verifies correctly."""
    pwd = "CorrectHorseBatteryStaple123!"
    h = hash_password(pwd)
    
    assert h.startswith("$argon2id$")
    assert verify_password(pwd, h) is True
    assert verify_password("WrongPassword123!", h) is False
    assert verify_password("", h) is False


def test_password_strength_validation():
    """Verify password length policies (8-128 chars)."""
    # Too short
    ok, err = validate_password_strength("short7")
    assert ok is False
    assert "at least 8" in err

    # Empty or whitespace
    ok, err = validate_password_strength("     ")
    assert ok is False

    # Valid
    ok, err = validate_password_strength("ValidPassword123!")
    assert ok is True
    assert err is None

    # Too long (> 128)
    long_pwd = "A" * 129
    ok, err = validate_password_strength(long_pwd)
    assert ok is False
    assert "not exceed 128" in err


def test_secure_tokens_and_digests():
    """Verify cryptographic tokens and SHA-256 constant-time verification."""
    token = generate_secure_token(32)
    assert len(token) > 30

    digest = hash_token(token)
    assert len(digest) == 64  # SHA-256 hex string

    assert verify_token_hash(token, digest) is True
    assert verify_token_hash("tampered_token", digest) is False


def test_email_normalization():
    """Verify emails are normalized to stripped lowercase."""
    assert normalize_email("  Test.User@CharityChain.ORG  ") == "test.user@charitychain.org"
    assert normalize_email("") == ""

