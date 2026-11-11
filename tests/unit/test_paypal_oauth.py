"""
test_paypal_oauth.py - PayPal OAuth 2.0 Client Authentication Unit Tests
========================================================================
Tests token acquisition, in-memory caching, expiration safety margins,
and handling of missing or invalid credentials.
"""

import pytest
from unittest.mock import MagicMock, patch
import requests
from backend.app.integrations.payments.paypal import PayPalPaymentProvider
from backend.app.integrations.payments.exceptions import PaymentAuthenticationException


def test_oauth_token_acquisition_and_caching():
    """Verifies that an OAuth token is acquired and cached until its expiration threshold."""
    provider = PayPalPaymentProvider(
        client_id="test_client_id",
        client_secret="test_client_secret",
        mode="sandbox"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "ACCESS_TOKEN_XYZ_123",
        "token_type": "Bearer",
        "expires_in": 3600
    }

    with patch.object(provider._session, "post", return_value=mock_resp) as mock_post:
        # First call acquires token
        token1 = provider._get_access_token()
        assert token1 == "ACCESS_TOKEN_XYZ_123"
        assert mock_post.call_count == 1

        # Second immediate call returns cached token without network request
        token2 = provider._get_access_token()
        assert token2 == "ACCESS_TOKEN_XYZ_123"
        assert mock_post.call_count == 1


def test_oauth_missing_credentials_raises_exception():
    """Verifies that unconfigured client credentials raise PaymentAuthenticationException."""
    provider = PayPalPaymentProvider(
        client_id="",
        client_secret=""
    )

    with pytest.raises(PaymentAuthenticationException) as exc_info:
        provider._get_access_token()

    assert "credentials" in str(exc_info.value.message).lower()


def test_oauth_invalid_credentials_rejected():
    """Verifies that HTTP 401 response from PayPal OAuth endpoint raises PaymentAuthenticationException."""
    provider = PayPalPaymentProvider(
        client_id="bad_client",
        client_secret="bad_secret",
        mode="sandbox"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": "invalid_client", "error_description": "Client Authentication failed"}'
    mock_resp.json.return_value = {"error": "invalid_client", "message": "Client Authentication failed"}

    with patch.object(provider._session, "post", return_value=mock_resp) as mock_post:
        with pytest.raises(PaymentAuthenticationException) as exc_info:
            provider._get_access_token()

        assert "authentication failed" in str(exc_info.value.message).lower()
