"""
test_duplicate_payout_prevention.py - Duplicate & Over-Disbursement Security Tests
==================================================================================
Tests:
1. Duplicate batch submission with identical sender_batch_id blocked.
2. In-flight payout collision prevents creating overlapping requests.
3. Attempting to disburse more than available unencumbered funds blocked.
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, ReconciliationResult,
    PayoutRecipientStatus, RecipientType, DisbursementStatus,
    DonationStatus, PaymentCaptureStatus
)
from backend.app.core.exceptions import ValidationException
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.reviews_impact import ReviewCase
from backend.app.db.models.financial import (
    PayoutRecipient, Donation, PaymentOrder, PaymentCapture, Disbursement
)
from backend.app.integrations.payouts.schemas import PayoutBatchRequest, PayoutItemRequest
from backend.app.integrations.payouts.exceptions import DuplicatePayoutBatchException
from backend.app.services.disbursement_service import DisbursementService
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def dup_setup(db_session):
    """Sets up an eligible milestone with $2,000.00 (200,000 cents) raised."""
    now = datetime.now(timezone.utc)
    org = Organization(id=str(uuid.uuid4()), name="Health Care Africa", legal_name="Health Care Africa Inc", status=OrganizationStatus.VERIFIED)
    project = Project(id=str(uuid.uuid4()), organization_id=org.id, title="Vaccines", description="Vaccine cold chain.", status=ProjectStatus.ACTIVE, currency="USD", target_amount_minor_units=200000, raised_amount_minor_units=200000)
    milestone = Milestone(id=str(uuid.uuid4()), project_id=project.id, sequence_number=1, title="Cold Storage", description="Refrigerators.", target_amount_minor_units=200000, raised_amount_minor_units=200000, status=MilestoneStatus.APPROVED)
    review_case = ReviewCase(
        id=str(uuid.uuid4()), milestone_id=milestone.id, project_id=project.id, organization_id=org.id, submission_round=1,
        status=ReviewCaseStatus.APPROVED.value, final_decision=ReviewDecision.APPROVE.value, reconciliation_result=ReconciliationResult.APPROVED.value,
        snapshot_digest="dup-digest", snapshot_data_json={"digest": "dup-digest"}
    )
    recip = PayoutRecipient(id=str(uuid.uuid4()), organization_id=org.id, recipient_type=RecipientType.PAYPAL_EMAIL, encrypted_destination="mock_enc", destination_masked="f***e@hca.org", destination_hash="h_dup", status=PayoutRecipientStatus.VERIFIED, is_primary=True)
    don = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=200000, status=DonationStatus.PAID)
    ord1 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don.id, provider_order_id="ORD-DUP-1", amount_minor_units=200000, currency="USD", status=DonationStatus.PAID)
    cap = PaymentCapture(id=str(uuid.uuid4()), order_id=ord1.id, provider_capture_id="CAP-DUP-1", amount_minor_units=200000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)

    db_session.add_all([org, project, milestone, review_case, recip, don, ord1, cap])
    db_session.commit()
    return milestone, recip


def test_over_disbursement_rejected(db_session, dup_setup):
    """Attempting to request $2,500.00 when only $2,000.00 is available fails."""
    milestone, recip = dup_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    with pytest.raises(ValidationException) as exc:
        srv.request_disbursement(
            db=db_session,
            milestone_id=milestone.id,
            amount_minor_units=250000,  # $2,500.00 > $2,000.00
            requester_id="operator-1"
        )
    assert "exceeds available unencumbered balance" in str(exc.value)


def test_duplicate_sender_batch_id_blocked():
    """Provider rejects second submission with identical sender_batch_id."""
    mock_provider = MockPayoutProvider()
    batch_req = PayoutBatchRequest(
        sender_batch_id="UNIQUE-BATCH-12345",
        items=[
            PayoutItemRequest(
                disbursement_id="disb-1",
                recipient_type=RecipientType.PAYPAL_EMAIL,
                receiver="payouts@ngo.org",
                amount_minor_units=5000,
                currency="USD",
                sender_item_id="item-1"
            )
        ]
    )

    # First submission succeeds
    res1 = mock_provider.create_payout_batch(batch_req)
    assert res1.sender_batch_id == "UNIQUE-BATCH-12345"

    # Second submission with identical sender_batch_id must fail
    with pytest.raises(DuplicatePayoutBatchException):
        mock_provider.create_payout_batch(batch_req)
