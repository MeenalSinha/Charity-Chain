"""
test_certificate_privacy.py - Unit Tests for Certificate Privacy & Redaction
=============================================================================
Verifies that public verification responses mask donor identity and contributions
unless explicit public attribution consent is granted, and never leaks donor email.
"""

import json
import uuid
import pytest
from backend.app.core.constants import CertificateStatus
from backend.app.db.models.reviews_impact import ImpactCertificate
from backend.app.services.public_verification_service import PublicVerificationService
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash


def test_anonymous_donor_masked_in_public_verification(db_session):
    """Verifies that when public_attribution is False, donor details are masked."""
    claims = {
        "donor": {
            "display_name": "Secret Philanthropist",
            "email": "secret@example.com"
        },
        "contribution": {
            "amount_cents": 250000,
            "currency": "USD"
        }
    }
    canonical_str = canonicalize_json(claims)
    payload_hash = compute_canonical_hash(claims)

    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=str(uuid.uuid4()),
        milestone_id=str(uuid.uuid4()),
        certificate_number="CC-PRIV-001",
        verification_id=f"v_id_{uuid.uuid4().hex[:16]}",
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Secret Philanthropist",
        public_attribution=False,  # Private
        canonical_payload=canonical_str,
        payload_hash=payload_hash,
        signature="mock_sig",
        signing_key_id="key-1"
    )
    db_session.add(cert)
    db_session.commit()

    result = PublicVerificationService.verify_certificate(db=db_session, verification_id=cert.verification_id)

    assert result["donor"]["display_name"] == "Anonymous Contributor"
    assert result["donor"]["is_anonymous"] is True
    assert result["contribution"]["attribution"] == "Private Contribution"
    assert "amount_cents" not in result["contribution"]
    assert "secret@example.com" not in json.dumps(result)


def test_consenting_donor_shown_in_public_verification(db_session):
    """Verifies that when public_attribution is True, donor name is visible, but email is not exposed."""
    claims = {
        "donor": {
            "display_name": "Open Philanthropist",
            "email": "open@example.com"
        },
        "contribution": {
            "amount_cents": 5000,
            "currency": "USD"
        }
    }
    canonical_str = canonicalize_json(claims)
    payload_hash = compute_canonical_hash(claims)

    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=str(uuid.uuid4()),
        milestone_id=str(uuid.uuid4()),
        certificate_number="CC-PRIV-002",
        verification_id=f"v_id_{uuid.uuid4().hex[:16]}",
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Open Philanthropist",
        public_attribution=True,  # Public
        canonical_payload=canonical_str,
        payload_hash=payload_hash,
        signature="mock_sig",
        signing_key_id="key-1"
    )
    db_session.add(cert)
    db_session.commit()

    result = PublicVerificationService.verify_certificate(db=db_session, verification_id=cert.verification_id)

    assert result["donor"]["display_name"] == "Open Philanthropist"
    assert result["contribution"]["amount_cents"] == 5000
    # Donor email must NEVER leak into public verification output
    assert "open@example.com" not in json.dumps(result)

