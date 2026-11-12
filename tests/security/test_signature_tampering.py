"""
test_signature_tampering.py - Security Tests for Cryptographic Tamper Resistance
================================================================================
Verifies that any modification to signed certificate claims, signatures, or key identifiers
is immediately detected and rejected by the verification engine.
"""

import json
import uuid
import pytest
from datetime import datetime

from backend.app.core.constants import CertificateStatus
from backend.app.db.models.reviews_impact import ImpactCertificate
from backend.app.core.canonical_json import canonicalize_json
from backend.app.services.key_management_service import KeyManagementService
from backend.app.services.public_verification_service import PublicVerificationService


@pytest.fixture
def genuine_certificate(db_session):
    kid, priv = KeyManagementService.get_active_private_key()

    claims = {
        "iss": "https://charitychain.org",
        "sub": "donation-sec-1",
        "certificate_number": "CC-SEC-001",
        "verification_id": f"v_id_{uuid.uuid4().hex[:16]}",
        "issued_at": datetime.utcnow().isoformat(),
        "donor": {"display_name": "Authentic Donor", "is_anonymous": False},
        "contribution": {"amount_cents": 5000, "currency": "USD"},
        "project": {"title": "Wildlife Shelter"},
        "milestone": {"title": "Solar Roof"}
    }
    canonical_str = canonicalize_json(claims)
    sig_b64, active_kid, payload_hash = KeyManagementService.sign_payload(canonical_str)

    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=str(uuid.uuid4()),
        milestone_id=str(uuid.uuid4()),
        certificate_number=claims["certificate_number"],
        verification_id=claims["verification_id"],
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Authentic Donor",
        public_attribution=True,
        canonical_payload=canonical_str,
        payload_hash=payload_hash,
        signature=sig_b64,
        signing_key_id=active_kid,
        signature_algorithm="Ed25519",
        issuer_id="https://charitychain.org"
    )
    db_session.add(cert)
    db_session.commit()
    db_session.refresh(cert)
    return cert


def test_tampered_payload_claims_detected(db_session, genuine_certificate):
    """Verifies that an attacker altering claims in the database is detected as PAYLOAD_MODIFIED."""
    # Attacker alters the contribution amount in the database
    claims_dict = json.loads(genuine_certificate.canonical_payload)
    claims_dict["contribution"]["amount_cents"] = 500000000  # Inflate donation
    tampered_json = json.dumps(claims_dict)

    genuine_certificate.canonical_payload = tampered_json
    db_session.commit()

    result = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=genuine_certificate.verification_id
    )

    assert result["is_payload_intact"] is False
    assert result["verification_outcome"] == "PAYLOAD_MODIFIED"


def test_forged_signature_rejected(db_session, genuine_certificate):
    """Verifies that replacing the digital signature with a forged signature fails validation."""
    # Generate an unrelated signature from a rogue key
    rogue_priv, _, _, _ = KeyManagementService.generate_keypair()
    forged_sig = rogue_priv.sign(genuine_certificate.canonical_payload.encode("utf-8"))
    import base64
    forged_sig_b64 = base64.b64encode(forged_sig).decode("utf-8")

    genuine_certificate.signature = forged_sig_b64
    db_session.commit()

    result = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=genuine_certificate.verification_id
    )

    assert result["is_signature_valid"] is False
    assert result["verification_outcome"] == "INVALID_SIGNATURE"


def test_forged_signing_key_id_rejected(db_session, genuine_certificate):
    """Verifies that setting a non-existent or rogue signing_key_id fails signature verification."""
    genuine_certificate.signing_key_id = "key-rogue-attacker-999"
    db_session.commit()

    result = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=genuine_certificate.verification_id
    )

    assert result["is_signature_valid"] is False
    assert result["verification_outcome"] == "INVALID_SIGNATURE"

