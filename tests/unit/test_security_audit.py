"""
test_security_audit.py - Unit Tests for Tamper-Evident Security Audit Trail
===========================================================================
Validates cryptographic hash chaining, append-only persistence,
sequential chain verification, and tamper detection.
"""

import pytest
import uuid
from backend.app.db.models.security_audit import (
    SecurityAuditEvent,
    compute_security_audit_hash,
    GENESIS_SECURITY_AUDIT_HASH,
)
from backend.app.services.security_audit_service import SecurityAuditService


def test_record_first_event_links_to_genesis(db_session):
    """Verifies that the first recorded event links directly to the genesis hash block."""
    event = SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOGIN_SUCCESS",
        action="user_login",
        severity="INFO",
        user_id="user-123",
        actor_email="donor@example.org",
        details={"auth_method": "password"}
    )

    assert event.id is not None
    assert event.previous_hash == GENESIS_SECURITY_AUDIT_HASH
    assert len(event.event_hash) == 64
    assert event.severity == "INFO"


def test_sequential_events_form_unbroken_hash_chain(db_session):
    """Verifies that subsequent events link to the previous event's hash."""
    e1 = SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOGIN_SUCCESS",
        action="user_login",
        severity="INFO",
        details={"step": 1}
    )

    e2 = SecurityAuditService.record_event(
        db=db_session,
        event_type="ROLE_CHANGED",
        action="escalate_role",
        severity="WARNING",
        details={"step": 2}
    )

    assert e2.previous_hash == e1.event_hash

    # Verify chain integrity
    report = SecurityAuditService.verify_chain_integrity(db_session)
    assert report["is_valid"] is True
    assert report["audited_count"] >= 2


def test_tamper_detection_flags_altered_payload(db_session):
    """Verifies that modifying an audit record's payload breaks chain integrity."""
    e1 = SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOCKOUT",
        action="account_lockout",
        severity="ERROR",
        details={"failed_attempts": 5}
    )

    # Malicious actor tampers with details directly in database
    e1.details = {"failed_attempts": 1}
    db_session.commit()

    # Integrity verification must detect discrepancy
    report = SecurityAuditService.verify_chain_integrity(db_session)
    assert report["is_valid"] is False
    assert report["broken_at_event_id"] == e1.event_id
    assert report["reason"] == "EVENT_HASH_MISMATCH"


def test_tamper_detection_flags_broken_link(db_session):
    """Verifies that tampering with previous_hash breaks chain integrity."""
    e1 = SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOGIN_SUCCESS",
        action="login_1",
        details={}
    )
    e2 = SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOGIN_SUCCESS",
        action="login_2",
        details={}
    )

    # Malicious actor modifies previous_hash
    e2.previous_hash = "f" * 64
    db_session.commit()

    report = SecurityAuditService.verify_chain_integrity(db_session)
    assert report["is_valid"] is False
    assert report["broken_at_event_id"] == e2.event_id
