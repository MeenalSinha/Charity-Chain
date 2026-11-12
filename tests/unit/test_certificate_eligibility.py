"""
test_certificate_eligibility.py - Unit Tests for Certificate Eligibility Service
================================================================================
Verifies that Proof-of-Impact certificates are issued only when genuine payment capture,
verified organization, dual-key human review approval, and clean dispute status exist.
"""

import uuid
import pytest
from datetime import datetime, timezone

from backend.app.core.constants import (
    DonationStatus, ProjectStatus, MilestoneStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, DisputeStatus, CertificateStatus
)
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture, PaymentRefund
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.reviews_impact import ReviewCase, ReviewDispute, ImpactCertificate
from backend.app.services.certificate_eligibility_service import CertificateEligibilityService


@pytest.fixture
def cert_setup(db_session):
    """Sets up an eligible donation and milestone environment."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Oceans Foundation",
        legal_name="Clean Oceans Global Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Ocean Cleanup Initiative",
        description="Cleaning plastic waste from coastal waters",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=100000,
        raised_amount_minor_units=100000
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Barrier Deployment",
        description="Deployment of 500m floating ocean barriers",
        status=MilestoneStatus.APPROVED,
        target_amount_minor_units=100000,
        raised_amount_minor_units=100000
    )
    db_session.add(milestone)

    donation = Donation(
        id=str(uuid.uuid4()),
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=10000,
        currency="USD",
        status=DonationStatus.PAID.value
    )
    db_session.add(donation)

    order = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=donation.id,
        amount_minor_units=10000,
        currency="USD",
        status="COMPLETED"
    )
    db_session.add(order)

    capture = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=order.id,
        amount_minor_units=10000,
        currency="USD",
        status="COMPLETED",
        captured_at=datetime.utcnow()
    )
    db_session.add(capture)

    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        snapshot_digest="a" * 64
    )
    db_session.add(review_case)
    db_session.commit()

    return {
        "org": org,
        "project": project,
        "milestone": milestone,
        "donation": donation,
        "order": order,
        "capture": capture,
        "review_case": review_case
    }


def test_fully_eligible_donation_passes(db_session, cert_setup):
    """Verifies that an eligible donation and approved milestone pass all gates."""
    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is True
    assert len(result.reasons) == 0
    assert result.qualifying_amount_cents == 10000
    assert result.review_case_id == cert_setup["review_case"].id


def test_reject_unconfirmed_payment(db_session, cert_setup):
    """Verifies rejection when no completed payment capture exists."""
    cert_setup["capture"].status = "PENDING"
    cert_setup["donation"].status = DonationStatus.PAYMENT_PENDING.value
    db_session.commit()

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is False
    assert "NO_CONFIRMED_CAPTURE" in result.reasons


def test_reject_fully_refunded_donation(db_session, cert_setup):
    """Verifies rejection when contribution is fully refunded."""
    refund = PaymentRefund(
        id=str(uuid.uuid4()),
        capture_id=cert_setup["capture"].id,
        amount_minor_units=10000,
        currency="USD",
        status="COMPLETED",
        refunded_at=datetime.utcnow()
    )
    db_session.add(refund)
    db_session.commit()

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is False
    assert "FULLY_REFUNDED_OR_ZERO_BALANCE" in result.reasons


def test_reject_unapproved_milestone(db_session, cert_setup):
    """Verifies rejection if milestone has not completed human review approval."""
    cert_setup["review_case"].final_decision = ReviewDecision.REJECT.value
    db_session.commit()

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is False
    assert "MILESTONE_NOT_HUMAN_APPROVED" in result.reasons


def test_reject_open_dispute(db_session, cert_setup):
    """Verifies rejection when milestone is currently subject to an unresolved appeal or dispute."""
    dispute = ReviewDispute(
        id=str(uuid.uuid4()),
        review_case_id=cert_setup["review_case"].id,
        milestone_id=cert_setup["milestone"].id,
        project_id=cert_setup["project"].id,
        organization_id=cert_setup["org"].id,
        dispute_category="EVIDENCE_MISINTERPRETATION",
        dispute_reason="Challenging reviewer decision",
        status=DisputeStatus.OPEN.value
    )
    db_session.add(dispute)
    db_session.commit()

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is False
    assert "UNRESOLVED_DISPUTE_EXISTS" in result.reasons


def test_reject_duplicate_active_certificate(db_session, cert_setup):
    """Verifies idempotency gate: rejects issuance if active certificate already exists."""
    existing_cert = ImpactCertificate(
        donation_id=cert_setup["donation"].id,
        milestone_id=cert_setup["milestone"].id,
        certificate_number="CC-CERT-EXISTING-001",
        verification_id="existing_v_id",
        idempotency_key="existing_key",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Donor",
        canonical_payload="{}",
        payload_hash="0" * 64,
        signature="mock_sig",
        signing_key_id="key-test"
    )
    db_session.add(existing_cert)
    db_session.commit()

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=cert_setup["donation"].id
    )
    assert result.is_eligible is False
    assert "CERTIFICATE_ALREADY_ISSUED" in result.reasons
