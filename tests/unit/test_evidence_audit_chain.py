"""
test_evidence_audit_chain.py - Cryptographic Audit Chain Unit Tests
===================================================================
Tests deterministic SHA-256 hash chaining, genesis event creation,
continuous sequence verification, and tamper detection.
"""

from datetime import datetime
from backend.app.db.models.evidence import compute_audit_hash


def test_genesis_audit_event_hash():
    """Verify genesis event hashes deterministic payload chained to all-zeros."""
    genesis_prev = "0" * 64
    evidence_id = "test-ev-1"
    actor_id = "user-123"
    event_type = "UPLOADED"
    details = {"filename": "solar.jpg", "size": 1024}
    timestamp_iso = "2026-11-11T12:00:00"

    hash1 = compute_audit_hash(genesis_prev, evidence_id, actor_id, event_type, details, timestamp_iso)
    hash2 = compute_audit_hash(genesis_prev, evidence_id, actor_id, event_type, details, timestamp_iso)

    assert len(hash1) == 64
    assert hash1 == hash2  # Deterministic


def test_sequential_audit_event_chaining():
    """Verify that event n cryptographically binds to event n-1."""
    prev_hash_0 = "0" * 64
    ev_id = "ev-chain-1"
    
    # Event 1: Upload
    h1 = compute_audit_hash(prev_hash_0, ev_id, "u1", "UPLOADED", {"file": "doc1.pdf"}, "2026-11-11T12:00:00")
    
    # Event 2: Verified
    h2 = compute_audit_hash(h1, ev_id, "u2", "INTEGRITY_VERIFIED", {"status": True}, "2026-11-11T12:05:00")
    
    # Event 3: Version Created
    h3 = compute_audit_hash(h2, ev_id, "u1", "VERSION_CREATED", {"v": 2}, "2026-11-11T12:10:00")

    assert h1 != h2
    assert h2 != h3
    assert len(h1) == 64
    assert len(h2) == 64
    assert len(h3) == 64


def test_tamper_detection_in_audit_chain():
    """Verify modifying any audit parameter breaks hash verification."""
    prev = "0" * 64
    ev_id = "ev-tamper-check"
    ts = "2026-11-11T12:00:00"

    original_hash = compute_audit_hash(prev, ev_id, "actor-1", "UPLOADED", {"score": 90}, ts)

    # Tamper with actor
    tampered_actor = compute_audit_hash(prev, ev_id, "actor-hacker", "UPLOADED", {"score": 90}, ts)
    assert tampered_actor != original_hash

    # Tamper with details
    tampered_details = compute_audit_hash(prev, ev_id, "actor-1", "UPLOADED", {"score": 100}, ts)
    assert tampered_details != original_hash

    # Tamper with event type
    tampered_event = compute_audit_hash(prev, ev_id, "actor-1", "APPROVED", {"score": 90}, ts)
    assert tampered_event != original_hash

    # Tamper with timestamp
    tampered_time = compute_audit_hash(prev, ev_id, "actor-1", "UPLOADED", {"score": 90}, "2026-11-11T12:00:01")
    assert tampered_time != original_hash
