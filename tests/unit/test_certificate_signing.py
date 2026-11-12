"""
test_certificate_signing.py - Unit Tests for Cryptographic Signing & Key Management
=====================================================================================
Verifies Ed25519 asymmetric digital signing against RFC 8785 canonical payloads,
independent signature verification, tamper detection, and JWKS public key exports.
"""

import base64
import pytest
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash
from backend.app.services.key_management_service import KeyManagementService


def test_generate_ed25519_keypair():
    """Verifies generation of Ed25519 keypair and standard JWK output."""
    priv, priv_pem, pub_pem, jwk = KeyManagementService.generate_keypair(key_id="key-test-01")
    assert priv is not None
    assert "BEGIN PRIVATE KEY" in priv_pem
    assert "BEGIN PUBLIC KEY" in pub_pem
    assert jwk["kty"] == "OKP"
    assert jwk["crv"] == "Ed25519"
    assert jwk["kid"] == "key-test-01"
    assert jwk["use"] == "sig"
    assert jwk["alg"] == "EdDSA"
    assert len(jwk["x"]) > 20


def test_sign_and_verify_valid_canonical_payload():
    """Verifies that a signed canonical payload passes independent signature verification."""
    priv, priv_pem, pub_pem, jwk = KeyManagementService.generate_keypair(key_id="key-test-sign")
    KeyManagementService.set_key_override(priv, "key-test-sign")

    try:
        claims = {
            "iss": "https://charitychain.org",
            "sub": "donation-1234",
            "amount_minor_units": 5000,
            "currency": "USD",
            "status": "ISSUED"
        }
        canonical_str = canonicalize_json(claims)
        sig_b64, kid, payload_hash = KeyManagementService.sign_payload(canonical_str)

        assert kid == "key-test-sign"
        assert len(payload_hash) == 64
        assert len(sig_b64) > 40

        # Verify using PEM
        is_valid_pem = KeyManagementService.verify_signature(canonical_str, sig_b64, pub_pem)
        assert is_valid_pem is True

        # Verify using raw base64 from JWK
        is_valid_raw = KeyManagementService.verify_signature(canonical_str, sig_b64, jwk["x"])
        assert is_valid_raw is True
    finally:
        KeyManagementService.set_key_override(None, None)


def test_signature_fails_on_modified_payload():
    """Verifies that modifying any part of the signed payload invalidates the signature."""
    priv, priv_pem, pub_pem, _ = KeyManagementService.generate_keypair(key_id="key-test-tamper")
    KeyManagementService.set_key_override(priv, "key-test-tamper")

    try:
        original = {"amount_minor_units": 10000, "project": "Clean Water"}
        canonical_str = canonicalize_json(original)
        sig_b64, _, _ = KeyManagementService.sign_payload(canonical_str)

        tampered = {"amount_minor_units": 1000000, "project": "Clean Water"}
        tampered_canonical = canonicalize_json(tampered)

        is_valid = KeyManagementService.verify_signature(tampered_canonical, sig_b64, pub_pem)
        assert is_valid is False
    finally:
        KeyManagementService.set_key_override(None, None)


def test_signature_fails_on_corrupted_signature_bytes():
    """Verifies that corrupting the signature bytes causes verification to fail."""
    priv, _, pub_pem, _ = KeyManagementService.generate_keypair()
    KeyManagementService.set_key_override(priv)

    try:
        canonical_str = canonicalize_json({"test": "data"})
        sig_b64, _, _ = KeyManagementService.sign_payload(canonical_str)

        # Corrupt signature by modifying last few characters
        sig_bytes = bytearray(base64.b64decode(sig_b64))
        sig_bytes[5] = (sig_bytes[5] + 1) % 256
        corrupted_sig_b64 = base64.b64encode(sig_bytes).decode("utf-8")

        is_valid = KeyManagementService.verify_signature(canonical_str, corrupted_sig_b64, pub_pem)
        assert is_valid is False
    finally:
        KeyManagementService.set_key_override(None, None)


def test_signature_fails_with_wrong_public_key():
    """Verifies that verifying with a different keypair's public key fails."""
    priv1, _, _, _ = KeyManagementService.generate_keypair(key_id="key-1")
    _, _, pub_pem2, _ = KeyManagementService.generate_keypair(key_id="key-2")

    KeyManagementService.set_key_override(priv1, "key-1")
    try:
        canonical_str = canonicalize_json({"msg": "signed by key 1"})
        sig_b64, _, _ = KeyManagementService.sign_payload(canonical_str)

        # Verify with key 2's public key
        is_valid = KeyManagementService.verify_signature(canonical_str, sig_b64, pub_pem2)
        assert is_valid is False
    finally:
        KeyManagementService.set_key_override(None, None)

