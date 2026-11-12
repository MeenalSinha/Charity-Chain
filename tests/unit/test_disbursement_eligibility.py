"""
test_disbursement_eligibility.py - Milestone Disbursement Eligibility Unit Tests
=================================================================================
Tests all compliance, human review, cryptographic digest, dispute, and financial gates.
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, ReconciliationResult,
    DisputeStatus, DisputeCategory, PayoutRecipientStatus,
    RecipientType, DisbursementStatus, DonationStatus, PaymentCaptureStatus
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.reviews_impact import ReviewCase, ReviewDispute
from backend.app.db.models.financial import (
    PayoutRecipient, Donation, PaymentOrder, PaymentCapture, Disbursement
)
from backend.app.services.disbursement_eligibility_service import DisbursementEligibilityService


@pytest.fixture
def eligibility_fixture(db_session):
    """Sets up a fully eligible milestone environment."""
    now = datetime.now(timezone.utc)
    org = Organization(
        id=str(uuid.uuid4()),
        name="Solar Light Org",
        legal_name="Solar Light Org Global",
        status=OrganizationStatus.VERIFIED
    )
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Solar Lighting",
        description="Installing solar power in remote hospitals.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=500000,
        raised_amount_minor_units=500000
    )
    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Solar Panel Array",
        description="Solar panel installation and grid tie-in.",
        target_amount_minor_units=500000,
        raised_amount_minor_units=500000,
        status=MilestoneStatus.APPROVED
    )
    # Review Case in APPROVED state
    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        reconciliation_result=ReconciliationResult.APPROVED.value,
        snapshot_digest="mock-valid-digest-12345",
        snapshot_data_json={"digest": "mock-valid-digest-12345"}
    )
    # Verified Payout Recipient
    recip = PayoutRecipient(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        recipient_type=RecipientType.PAYPAL_EMAIL,
        encrypted_destination="mock_enc_dest",
        destination_masked="f***e@solarlight.org",
        destination_hash="hash_123",
        status=PayoutRecipientStatus.VERIFIED,
        is_primary=True
    )
    # Donor contribution & capture ($5,000.00 / 500,000 cents)
    don = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=500000, status=DonationStatus.PAID)
    ord1 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don.id, provider_order_id="ORD-ELIG-1", amount_minor_units=500000, currency="USD", status=DonationStatus.PAID)
    cap = PaymentCapture(id=str(uuid.uuid4()), order_id=ord1.id, provider_capture_id="CAP-ELIG-1", amount_minor_units=500000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)

    db_session.add_all([org, project, milestone, review_case, recip, don, ord1, cap])
    db_session.commit()

    return org, project, milestone, review_case, recip


def test_fully_eligible_milestone(db_session, eligibility_fixture):
    """Verifies that when all prerequisites are met, evaluation returns is_eligible=True."""
    org, project, milestone, review_case, recip = eligibility_fixture

    res = DisbursementEligibilityService.evaluate_eligibility(db_session, milestone.id, requested_amount_minor_units=500000)

    assert res["is_eligible"] is True
    assert len(res["failed_checks"]) == 0
    assert len(res["rejection_reasons"]) == 0
    assert res["available_balance_minor_units"] == 500000
    assert res["recipient_id"] == recip.id


def test_ineligible_due_to_unresolved_dispute(db_session, eligibility_fixture):
    """An open dispute on the milestone must block disbursement."""
    org, project, milestone, review_case, recip = eligibility_fixture

    dispute = ReviewDispute(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        review_case_id=review_case.id,
        raised_by_id="ngo-user",
        dispute_category=DisputeCategory.EVIDENCE_MISINTERPRETATION.value,
        dispute_reason="Reviewer misread technical report",
        status=DisputeStatus.UNDER_REVIEW.value
    )
    db_session.add(dispute)
    db_session.commit()

    res = DisbursementEligibilityService.evaluate_eligibility(db_session, milestone.id)

    assert res["is_eligible"] is False
    assert "NO_UNRESOLVED_DISPUTES" in res["failed_checks"]
    assert any("unresolved dispute" in r for r in res["rejection_reasons"])


def test_ineligible_due_to_unverified_recipient(db_session, eligibility_fixture):
    """If the NGO payout recipient is suspended or pending, disbursement is blocked."""
    org, project, milestone, review_case, recip = eligibility_fixture
    recip.status = PayoutRecipientStatus.PENDING_VERIFICATION
    db_session.commit()

    res = DisbursementEligibilityService.evaluate_eligibility(db_session, milestone.id)

    assert res["is_eligible"] is False
    assert "VERIFIED_PAYOUT_RECIPIENT_CONFIGURED" in res["failed_checks"]


def test_ineligible_due_to_in_flight_payout(db_session, eligibility_fixture):
    """An active in-flight disbursement prevents creating duplicate requests."""
    org, project, milestone, review_case, recip = eligibility_fixture

    existing_disb = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        organization_id=org.id,
        amount_minor_units=200000,
        currency="USD",
        status=DisbursementStatus.PENDING_APPROVAL
    )
    db_session.add(existing_disb)
    db_session.commit()

    res = DisbursementEligibilityService.evaluate_eligibility(db_session, milestone.id)

    assert res["is_eligible"] is False
    assert "NO_IN_FLIGHT_DISBURSEMENT" in res["failed_checks"]
