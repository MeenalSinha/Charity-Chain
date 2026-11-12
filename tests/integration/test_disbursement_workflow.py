"""
test_disbursement_workflow.py - End-to-End Milestone Disbursement Workflow Integration Tests
=============================================================================================
Tests full Maker-Checker disbursement lifecycle:
1. Eligibility verification
2. Maker request creation
3. Checker authorization
4. Ledger commitment posting
5. Outbound batch submission
6. Ledger execution posting
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, ReconciliationResult,
    PayoutRecipientStatus, RecipientType, DisbursementStatus,
    PlatformRole, DonationStatus, PaymentCaptureStatus
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization, User
from backend.app.db.models.reviews_impact import ReviewCase
from backend.app.db.models.financial import (
    PayoutRecipient, Donation, PaymentOrder, PaymentCapture, Disbursement, PayoutBatch
)
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.ledger_service import LedgerService
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def workflow_setup(db_session):
    """Sets up a complete verifiable project, milestone, review case, and users."""
    now = datetime.now(timezone.utc)
    maker = User(
        id=str(uuid.uuid4()), email="maker@charity.io", full_name="Maker Operator",
        hashed_password="mock", platform_role=PlatformRole.FINANCE_OPERATOR, is_verified=True
    )
    checker = User(
        id=str(uuid.uuid4()), email="checker@charity.io", full_name="Checker Approver",
        hashed_password="mock", platform_role=PlatformRole.FINANCE_APPROVER, is_verified=True
    )
    org = Organization(id=str(uuid.uuid4()), name="Water Well Hope", legal_name="Water Well Hope Inc", status=OrganizationStatus.VERIFIED)
    project = Project(id=str(uuid.uuid4()), organization_id=org.id, title="Water Wells", description="Boreholes in Kenya.", status=ProjectStatus.ACTIVE, currency="USD", target_amount_minor_units=400000, raised_amount_minor_units=400000)
    milestone = Milestone(id=str(uuid.uuid4()), project_id=project.id, sequence_number=1, title="Borehole Drilling", description="Drilling 2 boreholes.", target_amount_minor_units=400000, raised_amount_minor_units=400000, status=MilestoneStatus.APPROVED)
    review_case = ReviewCase(
        id=str(uuid.uuid4()), milestone_id=milestone.id, project_id=project.id, organization_id=org.id, submission_round=1,
        status=ReviewCaseStatus.APPROVED.value, final_decision=ReviewDecision.APPROVE.value, reconciliation_result=ReconciliationResult.APPROVED.value,
        snapshot_digest="wf-digest", snapshot_data_json={"digest": "wf-digest"}
    )
    recip = PayoutRecipient(
        id=str(uuid.uuid4()), organization_id=org.id, recipient_type=RecipientType.PAYPAL_EMAIL,
        encrypted_destination="mock_enc", destination_masked="p***t@waterwell.org", destination_hash="h_wf",
        status=PayoutRecipientStatus.VERIFIED, is_primary=True
    )
    don = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=400000, status=DonationStatus.PAID)
    ord1 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don.id, provider_order_id="ORD-WF-1", amount_minor_units=400000, currency="USD", status=DonationStatus.PAID)
    cap = PaymentCapture(id=str(uuid.uuid4()), order_id=ord1.id, provider_capture_id="CAP-WF-1", amount_minor_units=400000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)

    db_session.add_all([maker, checker, org, project, milestone, review_case, recip, don, ord1, cap])
    db_session.commit()

    return maker, checker, milestone, recip


def test_complete_maker_checker_and_batch_submission_workflow(db_session, workflow_setup):
    """Executes the entire milestone disbursement flow through batch settlement."""
    maker, checker, milestone, recip = workflow_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    # 1. Maker requests $3,000.00 disbursement
    disb = srv.request_disbursement(
        db=db_session,
        milestone_id=milestone.id,
        amount_minor_units=300000,
        requester_id=maker.id,
        notes="Phase 1 completion evidence verified by human review team"
    )
    assert disb.status == DisbursementStatus.PENDING_APPROVAL

    # 2. Checker approves disbursement
    approved = srv.approve_disbursement(
        db=db_session,
        disbursement_id=disb.id,
        approver_id=checker.id,
        notes="All 9 eligibility gates confirmed"
    )
    assert approved.status == DisbursementStatus.APPROVED
    assert approved.approved_by_id == checker.id

    # Verify commitment journal entry was posted
    tb_comm = LedgerService.get_trial_balance(db_session)
    assert tb_comm["is_balanced"] is True

    # 3. Submit payout batch
    batch = srv.create_and_submit_payout_batch(
        db=db_session,
        disbursement_ids=[approved.id],
        submitter_id=checker.id,
        email_subject="Water Well Project Milestone #1 Payout"
    )

    assert batch.status in (DisbursementStatus.SUCCEEDED, DisbursementStatus.PROCESSING)
    assert batch.total_amount_minor_units == 300000
    assert batch.provider_batch_id is not None

    db_session.refresh(disb)
    assert disb.status == DisbursementStatus.SUCCEEDED
    assert disb.disbursed_at is not None
    assert disb.provider_item_id is not None

    # Verify execution journal entry and trial balance
    tb_final = LedgerService.get_trial_balance(db_session)
    assert tb_final["is_balanced"] is True
    assert tb_final["discrepancy_minor_units"] == 0
