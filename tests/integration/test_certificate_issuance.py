"""
test_certificate_issuance.py - Integration Tests for Certificate Issuance Pipeline
===================================================================================
Verifies end-to-end issuance of cryptographic Proof-of-Impact certificates,
including eligibility validation, Ed25519 signing, PDF generation, and Impact Ledger recording.
"""

import uuid
import pytest
from datetime import datetime, timezone

from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, DonationStatus, PaymentCaptureStatus,
    PaymentOrderStatus, CertificateStatus, CertificateType, ImpactLedgerEventType
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization, User
from backend.app.db.models.reviews_impact import ReviewCase, ImpactCertificate, ImpactLedgerEvent
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture
from backend.app.services.certificate_service import CertificateService


@pytest.fixture
def issuance_fixture(db_session):
    now = datetime.now(timezone.utc)
    donor = User(
        id=str(uuid.uuid4()),
        email="donor@example.com",
        full_name="Sarah Connor",
        hashed_password="mock",
        is_verified=True
    )
    db_session.add(donor)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Amazon Reforestation",
        legal_name="Amazon Reforestation Alliance",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Canopy Protection 2026",
        description="Planting indigenous tree species across 500 hectares",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=500000,
        raised_amount_minor_units=500000
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Sapling Nursery Setup",
        description="Establishing nursery for 100,000 seedlings",
        target_amount_minor_units=500000,
        raised_amount_minor_units=500000,
        status=MilestoneStatus.APPROVED
    )
    db_session.add(milestone)

    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        snapshot_digest="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    db_session.add(review_case)

    donation = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=25000,
        currency="USD",
        status=DonationStatus.PAID
    )
    db_session.add(donation)

    order = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=donation.id,
        provider_order_id=f"ORD-{uuid.uuid4().hex[:8]}",
        amount_minor_units=25000,
        currency="USD",
        status=PaymentOrderStatus.COMPLETED
    )
    db_session.add(order)

    capture = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=order.id,
        provider_capture_id=f"CAP-{uuid.uuid4().hex[:8]}",
        amount_minor_units=25000,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED,
        captured_at=now
    )
    db_session.add(capture)
    db_session.commit()

    return {
        "donor": donor,
        "org": org,
        "project": project,
        "milestone": milestone,
        "review_case": review_case,
        "donation": donation,
        "order": order,
        "capture": capture
    }


def test_issue_contribution_certificate_success(db_session, issuance_fixture):
    """Verifies that an eligible donation produces a valid, cryptographically signed certificate."""
    cert = CertificateService.issue_contribution_certificate(
        db=db_session,
        donation_id=issuance_fixture["donation"].id,
        recipient_name="Sarah Connor",
        public_attribution=True,
        actor_id="admin-issuance"
    )

    assert cert is not None
    assert cert.certificate_type == CertificateType.CONTRIBUTION_IMPACT.value
    assert cert.status == CertificateStatus.ISSUED.value
    assert cert.certificate_number.startswith("CC-CERT-")
    assert len(cert.verification_id) >= 16
    assert cert.signature is not None
    assert len(cert.signature) > 40
    assert len(cert.payload_hash) == 64
    assert cert.signing_key_id is not None
    assert cert.pdf_storage_key is not None
    assert cert.pdf_hash is not None

    # Verify Impact Ledger stream event was recorded
    stream_id = f"milestone-{issuance_fixture['milestone'].id}"
    ledger_event = db_session.query(ImpactLedgerEvent).filter_by(
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.CERTIFICATE_ISSUED.value
    ).first()
    assert ledger_event is not None
    assert ledger_event.sequence_number >= 1


def test_reject_duplicate_issuance_idempotency(db_session, issuance_fixture):
    """Verifies that re-issuing a certificate for the same donation raises ValueError."""
    CertificateService.issue_contribution_certificate(
        db=db_session,
        donation_id=issuance_fixture["donation"].id,
        recipient_name="Sarah Connor"
    )

    with pytest.raises(ValueError, match="CERTIFICATE_ALREADY_ISSUED"):
        CertificateService.issue_contribution_certificate(
            db=db_session,
            donation_id=issuance_fixture["donation"].id,
            recipient_name="Sarah Connor"
        )
