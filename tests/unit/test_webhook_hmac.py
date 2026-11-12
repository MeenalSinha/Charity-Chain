"""
test_webhook_hmac.py - Unit Tests for Webhook HMAC Signatures & Replay Prevention
================================================================================
Verifies HMAC-SHA256 signature computation, verification, tolerance window,
and rejection of expired or tampered signatures.
"""

import time
import pytest
from backend.app.services.webhook_delivery_service import WebhookDeliveryService


def test_hmac_signature_generation_and_verification():
    """Verifies valid signature computation and verification with matching secret."""
    secret = "whsec_test_secret_key_1234567890abcdef"
    payload = '{"event_id": "evt-123", "event_type": "donation.captured"}'
    timestamp = int(time.time())

    sig_header = WebhookDeliveryService.compute_signature(payload, secret, timestamp)
    assert sig_header.startswith(f"t={timestamp},v1=")

    # Verification must succeed
    is_valid = WebhookDeliveryService.verify_signature(payload, sig_header, secret)
    assert is_valid is True


def test_hmac_signature_fails_on_tampered_payload():
    """Verifies that tampering with the payload invalidates the signature."""
    secret = "whsec_test_secret_key_1234567890abcdef"
    original_payload = '{"amount": 5000}'
    tampered_payload = '{"amount": 5000000}'
    timestamp = int(time.time())

    sig_header = WebhookDeliveryService.compute_signature(original_payload, secret, timestamp)
    is_valid = WebhookDeliveryService.verify_signature(tampered_payload, sig_header, secret)
    assert is_valid is False


def test_hmac_signature_fails_on_wrong_secret():
    """Verifies that verification fails with a different secret."""
    secret_a = "whsec_correct_secret"
    secret_b = "whsec_wrong_secret"
    payload = '{"event": "test"}'
    timestamp = int(time.time())

    sig_header = WebhookDeliveryService.compute_signature(payload, secret_a, timestamp)
    is_valid = WebhookDeliveryService.verify_signature(payload, sig_header, secret_b)
    assert is_valid is False


def test_hmac_signature_replay_prevention():
    """Verifies that timestamps older than tolerance window (300s) are rejected."""
    secret = "whsec_test_secret"
    payload = '{"event": "test"}'
    expired_timestamp = int(time.time()) - 350  # 350 seconds ago (> 300s tolerance)

    sig_header = WebhookDeliveryService.compute_signature(payload, secret, expired_timestamp)
    is_valid = WebhookDeliveryService.verify_signature(payload, sig_header, secret, tolerance_seconds=300)
    assert is_valid is False


def test_hmac_signature_malformed_header():
    """Verifies that malformed headers fail gracefully."""
    secret = "whsec_test_secret"
    payload = '{"event": "test"}'

    assert WebhookDeliveryService.verify_signature(payload, "invalid_header", secret) is False
    assert WebhookDeliveryService.verify_signature(payload, "t=abc,v1=def", secret) is False
    assert WebhookDeliveryService.verify_signature(payload, "", secret) is False

