"""
test_public_verification_api.py - Integration Tests for Public Verification & JWKS Endpoints
=============================================================================================
Tests public QR verification, cryptographic proof envelopes, and RFC 7517 JWKS discovery.
"""

import uuid
import pytest
from datetime import datetime

from backend.app.core.constants import CertificateStatus
from backend.app.db.models.reviews_impact import ImpactCertificate
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash
from backend.app.services.key_management_service import KeyManagementService
from backend.app.api.v1.certificates import (
    get_public_certificate_verification,
    get_public_certificate_proof,
    get_jwks_discovery as get_jwks
)


@pytest.fixture
def active_certificate(db_session):
    """Sets up an authentic, cryptographically signed active certificate."""
    kid, priv = KeyManagementService.get_active_private_key()

    claims = {
        "iss": "https://charitychain.org",
        "certificate_number": "CC-PUB-001",
        "verification_id": f"v_id_{uuid.uuid4().hex[:16]}",
        "issued_at": datetime.utcnow().isoformat(),
        "donor": {"display_name": "Jane Public", "is_anonymous": False},
        "contribution": {"amount_cents": 10000, "currency": "USD"},
        "project": {"title": "Solar Schools", "id": "proj-1"},
        "milestone": {"title": "Solar Array Installation", "id": "m-1"},
        "organization": {"name": "Clean Energy Initiative"},
        "verification_summary": {
            "evidence_count": 5,
            "human_approval": True,
            "dispute_count": 0
        }
    }
    canonical_str = canonicalize_json(claims)
    sig_b64, active_kid, payload_hash = KeyManagementService.sign_payload(canonical_str)

    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=str(uuid.uuid4()),
        milestone_id="m-1",
        project_id="proj-1",
        organization_id=str(uuid.uuid4()),
        certificate_number="CC-PUB-001",
        verification_id=claims["verification_id"],
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Jane Public",
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


def test_public_verification_endpoint_valid_certificate(db_session, active_certificate):
    """Verifies that an authentic certificate returns VALID outcome and cryptographic proof."""
    response = get_public_certificate_verification(
        verification_id=active_certificate.verification_id,
        db=db_session
    )

    assert response["verification_outcome"] == "VALID"
    assert response["is_signature_valid"] is True
    assert response["is_payload_intact"] is True
    assert response["certificate_status"] == CertificateStatus.ISSUED.value
    assert response["certificate_number"] == active_certificate.certificate_number
    assert response["donor"]["display_name"] == "Jane Public"
    assert response["contribution"]["amount_cents"] == 10000


def test_public_verification_unknown_certificate(db_session):
    """Verifies that querying a non-existent verification ID returns UNKNOWN_CERTIFICATE."""
    response = get_public_certificate_verification(
        verification_id="non-existent-v-id",
        db=db_session
    )

    assert response["verification_outcome"] == "UNKNOWN_CERTIFICATE"
    assert response["is_signature_valid"] is False
    assert response["certificate_status"] == "NOT_FOUND"


def test_public_proof_endpoint(db_session, active_certificate):
    """Verifies the standalone cryptographic proof envelope."""
    proof = get_public_certificate_proof(
        verification_id=active_certificate.verification_id,
        db=db_session
    )

    assert proof["verification_id"] == active_certificate.verification_id
    assert proof["signature_algorithm"] == "Ed25519"
    assert proof["payload_hash"] == active_certificate.payload_hash
    assert proof["canonical_payload"] == active_certificate.canonical_payload
    assert proof["signature"] == active_certificate.signature
    assert proof["jwks_uri"] == "/.well-known/jwks.json"


def test_jwks_discovery_endpoint(db_session):
    """Verifies RFC 7517 JWKS discovery endpoint response format."""
    jwks = get_jwks(db=db_session)

    assert "keys" in jwks
    assert isinstance(jwks["keys"], list)
    assert len(jwks["keys"]) >= 1

    key = jwks["keys"][0]
    assert key["kty"] == "OKP"
    assert key["crv"] == "Ed25519"
    assert key["use"] == "sig"
    assert key["alg"] == "EdDSA"
    assert "kid" in key
    assert "x" in key
