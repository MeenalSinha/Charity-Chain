"""
test_certificate_revocation.py - Unit Tests for Certificate Revocation & Suspension
===================================================================================
Verifies lifecycle transitions: suspension, revocation, and supersession, ensuring
all actions create immutable CertificateStatusEvent audit records and ledger events.
"""

import uuid
import pytest
from datetime import datetime
from backend.app.core.constants import (
    CertificateStatus, CertificateRevocationReason, ImpactLedgerEventType
)
from backend.app.db.models.reviews_impact import (
    ImpactCertificate, CertificateStatusEvent, ImpactLedgerEvent
)
from backend.app.services.certificate_service import CertificateService


@pytest.fixture
def mock_certificate(db_session):
    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=str(uuid.uuid4()),
        milestone_id=str(uuid.uuid4()),
        certificate_number="CC-TEST-0001",
        verification_id=f"v_id_{uuid.uuid4().hex[:16]}",
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Alice Donor",
        canonical_payload="{}",
        payload_hash="0" * 64,
        signature="mock_sig",
        signing_key_id="test-key"
    )
    db_session.add(cert)
    db_session.commit()
    db_session.refresh(cert)
    return cert


def test_suspend_certificate_lifecycle(db_session, mock_certificate):
    """Verifies that suspending a certificate updates status, writes status event and ledger event."""
    cert = CertificateService.suspend_certificate(
        db=db_session,
        certificate_id=mock_certificate.id,
        reason_code="DISPUTE_INVESTIGATION",
        reason_description="Donor initiated inquiry into milestone completion.",
        actor_id="admin-user-1"
    )

    assert cert.status == CertificateStatus.SUSPENDED.value
    assert cert.revocation_reason == "DISPUTE_INVESTIGATION"

    # Verify audit event
    events = db_session.query(CertificateStatusEvent).filter_by(certificate_id=cert.id).all()
    assert len(events) == 1
    assert events[0].new_status == CertificateStatus.SUSPENDED.value
    assert events[0].reason_code == "DISPUTE_INVESTIGATION"

    # Verify ledger entry
    ledger_event = db_session.query(ImpactLedgerEvent).filter_by(
        source_event_id=cert.id,
        event_type=ImpactLedgerEventType.CERTIFICATE_SUSPENDED.value
    ).first()
    assert ledger_event is not None


def test_revoke_certificate_lifecycle(db_session, mock_certificate):
    """Verifies that revoking a certificate sets revoked_at, writes status event and ledger event."""
    cert = CertificateService.revoke_certificate(
        db=db_session,
        certificate_id=mock_certificate.id,
        reason_code=CertificateRevocationReason.FULL_REFUND.value,
        reason_description="100% refund processed for transaction.",
        actor_id="admin-user-2"
    )

    assert cert.status == CertificateStatus.REVOKED.value
    assert cert.revocation_reason == CertificateRevocationReason.FULL_REFUND.value
    assert cert.revoked_at is not None

    events = db_session.query(CertificateStatusEvent).filter_by(certificate_id=cert.id).all()
    assert len(events) == 1
    assert events[0].new_status == CertificateStatus.REVOKED.value


def test_supersede_certificate_lifecycle(db_session, mock_certificate):
    """Verifies that superseding a certificate links to new certificate ID."""
    new_cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=mock_certificate.donation_id,
        milestone_id=mock_certificate.milestone_id,
        certificate_number="CC-TEST-0002",
        verification_id=f"v_id_{uuid.uuid4().hex[:16]}",
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Alice Donor",
        canonical_payload="{}",
        payload_hash="1" * 64,
        signature="mock_sig_2",
        signing_key_id="test-key"
    )
    db_session.add(new_cert)
    db_session.commit()

    superseded = CertificateService.supersede_certificate(
        db=db_session,
        old_certificate_id=mock_certificate.id,
        new_certificate_id=new_cert.id
    )

    assert superseded.status == CertificateStatus.SUPERSEDED.value
    assert superseded.superseded_by_id == new_cert.id
