"""
test_domain_events.py - Unit Tests for Domain Events & Canonical Serialization
=============================================================================
Tests canonical domain event construction, RFC 8785 serialization, and
automatic PII and secret stripping from event payloads.
"""

import json
import pytest
from backend.app.core.events import DomainEvent, sanitize_payload_for_event
from backend.app.core.constants import DomainEventType


def test_domain_event_creation_and_canonical_json():
    """Verifies domain event fields and deterministic canonical JSON representation."""
    event = DomainEvent.create(
        event_type=DomainEventType.DONATION_CREATED.value,
        entity_type="donation",
        entity_id="don-123",
        organization_id="org-456",
        actor_user_id="user-789",
        payload={
            "b_key": "val_b",
            "a_key": "val_a",
            "amount_minor_units": 5000,
            "currency": "USD"
        }
    )

    assert event.event_type == DomainEventType.DONATION_CREATED.value
    assert event.entity_type == "donation"
    assert event.entity_id == "don-123"
    assert "1.0" in event.schema_version
    assert event.id is not None
    assert event.timestamp is not None

    canonical_str = event.to_canonical_json()
    assert isinstance(canonical_str, str)
    # RFC 8785 keys should be sorted deterministically
    parsed = json.loads(canonical_str)
    assert parsed["entity_id"] == "don-123"
    assert parsed["data"]["a_key"] == "val_a"


def test_pii_and_credential_sanitization():
    """Verifies that sensitive tokens, passwords, and authorization data are recursively stripped."""
    dirty_payload = {
        "user_id": "u-001",
        "email": "donor@example.com",
        "password": "SuperSecretPassword123!",
        "access_token": "bearer-token-xyz",
        "api_key": "paypal-secret-999",
        "headers": {
            "Authorization": "Bearer sensitive-token",
            "Content-Type": "application/json"
        },
        "nested": {
            "client_secret": "donotlog",
            "safe_counter": 42
        }
    }

    clean_payload = sanitize_payload_for_event(dirty_payload)

    assert clean_payload["user_id"] == "u-001"
    assert clean_payload["email"] == "donor@example.com"
    assert clean_payload["password"] == "[REDACTED]"
    assert clean_payload["access_token"] == "[REDACTED]"
    assert clean_payload["api_key"] == "[REDACTED]"
    assert clean_payload["headers"]["Authorization"] == "[REDACTED]"
    assert clean_payload["headers"]["Content-Type"] == "application/json"
    assert clean_payload["nested"]["client_secret"] == "[REDACTED]"
    assert clean_payload["nested"]["safe_counter"] == 42


def test_domain_event_hash_generation():
    """Verifies that SHA-256 hash calculation is deterministic for the exact same event content."""
    event = DomainEvent.create(
        event_type="test.event",
        entity_type="test",
        entity_id="t-1",
        payload={"message": "hello world"}
    )

    hash1 = event.compute_sha256()
    hash2 = event.compute_sha256()
    assert hash1 == hash2
    assert len(hash1) == 64
