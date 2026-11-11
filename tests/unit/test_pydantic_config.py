"""
test_pydantic_config.py - Tests for Pydantic Settings & Zero-Hardcoding Enforcement
"""

import pytest
from pydantic import ValidationError
from backend.app.core.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.ENVIRONMENT in ["development", "testing", "staging", "production"]
    assert s.API_PORT > 0
    assert s.VERIFICATION_CONFIDENCE_THRESHOLD == 0.65
    assert s.TAMPER_ELA_THRESHOLD == 12.0
    assert len(s.SUPPORTED_CURRENCIES) >= 3


def test_production_security_enforcement():
    # Production with insecure default key must fail validation
    with pytest.raises(ValueError, match="Insecure default SECRET_KEY"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="development_insecure_secret_key_change_in_production_1234567890"
        )
    
    # Production with DEBUG=True must fail validation
    with pytest.raises(ValueError, match="DEBUG mode must be disabled"):
        Settings(
            ENVIRONMENT="production",
            DEBUG=True,
            SECRET_KEY="a_very_secure_and_long_random_production_secret_key_here_999"
        )
    
    # Valid production settings
    prod = Settings(
        ENVIRONMENT="production",
        DEBUG=False,
        SECRET_KEY="a_very_secure_and_long_random_production_secret_key_here_999"
    )
    assert prod.ENVIRONMENT == "production"
    assert prod.DEBUG is False

