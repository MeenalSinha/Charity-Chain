"""
test_exceptions_logging.py - Tests for Exceptions and Sanitized Logging
"""

import pytest
from backend.app.core.exceptions import (
    CharityChainException, EntityNotFoundError,
    FinancialValidationError, ProviderUnavailableError
)
from backend.app.core.logging import sanitize_data


def test_exception_status_codes():
    nf = EntityNotFoundError("Project", "123")
    assert nf.status_code == 404
    assert nf.code == "ENTITY_NOT_FOUND"
    
    fin = FinancialValidationError("Negative amount prohibited")
    assert fin.status_code == 422
    
    prov = ProviderUnavailableError("PayPal", "Gateway connection timed out")
    assert prov.status_code == 503


def test_sanitize_data_masks_secrets():
    payload = {
        "username": "alice",
        "password": "secret_password_123",
        "api_secret": "xyz_live_secret",
        "details": {
            "token": "bearer_jwt_token_456",
            "public_key": "pub_key_ok"
        }
    }
    cleaned = sanitize_data(payload)
    assert cleaned["username"] == "alice"
    assert cleaned["password"] == "***REDACTED***"
    assert cleaned["api_secret"] == "***REDACTED***"
    assert cleaned["details"]["token"] == "***REDACTED***"
    assert cleaned["details"]["public_key"] == "pub_key_ok"

