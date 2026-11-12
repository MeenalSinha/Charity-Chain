"""
test_impact_ledger_hashing.py - Unit Tests for Canonical JSON & Ledger Hash Chaining
====================================================================================
Tests RFC 8785 JSON canonicalization, float prohibition, deterministic SHA-256 digests,
and the cryptographic hash chaining formula of the append-only Impact Ledger.
"""

import hashlib
import pytest
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash
from backend.app.services.impact_ledger_service import ImpactLedgerService, GENESIS_HASH


def test_canonical_json_key_sorting():
    """Verifies that dictionaries with different key orders produce identical canonical strings."""
    data1 = {"beta": "two", "alpha": "one", "gamma": 3}
    data2 = {"alpha": "one", "gamma": 3, "beta": "two"}

    c1 = canonicalize_json(data1)
    c2 = canonicalize_json(data2)

    assert c1 == c2
    assert c1 == '{"alpha":"one","beta":"two","gamma":3}'


def test_canonical_json_no_extraneous_whitespace():
    """Verifies that canonical JSON contains no whitespace outside strings."""
    data = {"nested": {"arr": [1, 2, 3], "flag": True}}
    c = canonicalize_json(data)
    assert " " not in c
    assert c == '{"nested":{"arr":[1,2,3],"flag":true}}'


def test_canonical_json_float_rejection():
    """Verifies that floating-point numbers are strictly forbidden to prevent precision drift."""
    with pytest.raises(ValueError, match="Floating-point values.*are forbidden"):
        canonicalize_json({"amount": 10.50})

    with pytest.raises(ValueError, match="Floating-point values.*are forbidden"):
        canonicalize_json([1, 2, 3.14])


def test_compute_canonical_hash_consistency():
    """Verifies deterministic SHA-256 computation on canonical data structures."""
    data = {"claim": "impact_verified", "milestone_id": "m-42"}
    digest = compute_canonical_hash(data)
    expected = hashlib.sha256('{"claim":"impact_verified","milestone_id":"m-42"}'.encode("utf-8")).hexdigest()
    assert digest == expected
    assert len(digest) == 64


def test_impact_ledger_event_hash_chaining_formula():
    """Verifies that compute_event_hash follows the exact domain-separated formula."""
    prev_hash = GENESIS_HASH
    stream_id = "milestone-101"
    sequence_number = 1
    event_type = "MILESTONE_COMPLETED"
    payload_hash = hashlib.sha256(b"canonical_payload").hexdigest()

    expected_msg = f"{prev_hash}|{stream_id}|{sequence_number}|{event_type}|{payload_hash}"
    expected_hash = hashlib.sha256(expected_msg.encode("utf-8")).hexdigest()

    calculated_hash = ImpactLedgerService.compute_event_hash(
        previous_hash=prev_hash,
        stream_id=stream_id,
        sequence_number=sequence_number,
        event_type=event_type,
        payload_hash=payload_hash
    )

    assert calculated_hash == expected_hash

