"""
security.py - Cryptography, Password Security & JWT Token Management
=====================================================================
Implements Argon2id password hashing, constant-time verification,
secure token generation, SHA-256 token digests, and strict JWT creation/decoding.
"""

import hmac
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from backend.app.core.config import settings
from backend.app.core.exceptions import CharityChainException

# Initialize Argon2id password hasher with secure defaults
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32
)


class AuthenticationException(CharityChainException):
    """Raised when authentication credentials or token validation fails."""
    def __init__(self, message: str = "Authentication failed.", code: str = "AUTHENTICATION_FAILED", status_code: int = 401, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=code, status_code=status_code, details=details)


class TokenExpiredException(AuthenticationException):
    """Raised when a JWT or security token has passed its expiration timestamp."""
    def __init__(self, message: str = "Token has expired. Please authenticate again.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="TOKEN_EXPIRED", status_code=401, details=details)


class InvalidTokenException(AuthenticationException):
    """Raised when token signature, claims, or format are invalid."""
    def __init__(self, message: str = "Invalid authentication token.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="INVALID_TOKEN", status_code=401, details=details)


# ============================================================================
# PASSWORD HASHING & VALIDATION
# ============================================================================

def hash_password(password: str) -> str:
    """Generates an Argon2id salted password hash."""
    if not password:
        raise ValueError("Password cannot be empty.")
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2id hash using constant-time check."""
    if not plain_password or not hashed_password:
        return False
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validates password length and structure.
    Enforces minimum length of 8 and maximum of 128 characters without restrictive arbitrary rules.
    """
    if not password or len(password.strip()) == 0:
        return False, "Password cannot be empty or solely whitespace."
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long."
    if len(password) > settings.PASSWORD_MAX_LENGTH:
        return False, f"Password must not exceed {settings.PASSWORD_MAX_LENGTH} characters."
    return True, None


def normalize_email(email: str) -> str:
    """Normalizes an email address to lowercase stripped form."""
    if not email:
        return ""
    return email.strip().lower()


# ============================================================================
# CRYPTOGRAPHIC TOKENS & DIGESTS
# ============================================================================

def generate_secure_token(nbytes: int = 32) -> str:
    """Generates a cryptographically strong URL-safe random string."""
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Computes a SHA-256 hexadecimal digest of a plaintext security token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token_hash(token: str, token_hash: str) -> bool:
    """Performs constant-time comparison of a token against its stored SHA-256 digest."""
    if not token or not token_hash:
        return False
    computed = hash_token(token)
    return hmac.compare_digest(computed, token_hash)


# ============================================================================
# JWT ACCESS TOKEN MANAGEMENT
# ============================================================================

def create_access_token(
    subject: str,
    platform_role: str,
    session_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Creates a signed JWT access token containing subject, role, and standard claims.
    Enforces short expiration (default 15 minutes).
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(subject),
        "role": str(platform_role),
        "session_id": str(session_id) if session_id else None,
        "iss": "charitychain",
        "aud": "charitychain-api",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    token = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and verifies a JWT access token against server configuration.
    Validates signature, algorithm whitelist, issuer, audience, and expiration.
    """
    if not token:
        raise InvalidTokenException("Missing authentication token.")

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            issuer="charitychain",
            audience="charitychain-api",
            options={"require": ["exp", "iat", "sub", "jti"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except jwt.InvalidAlgorithmError as e:
        raise InvalidTokenException(f"Disallowed signing algorithm: {str(e)}")
    except jwt.InvalidTokenError as e:
        raise InvalidTokenException(f"Token validation failed: {str(e)}")

