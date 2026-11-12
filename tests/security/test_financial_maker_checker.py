"""
test_financial_maker_checker.py - Maker-Checker Financial Segregation of Duties Tests
====================================================================================
Tests strict segregation of duties:
1. The user who initiates a disbursement request cannot approve it.
2. Self-approval attempts are blocked with ForbiddenException.
3. An independent authorized finance approver is mandatory for authorization.
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
from backend.app.core.exceptions import ForbiddenException
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization, User
from backend.app.db.models.reviews_impact import ReviewCase
from backend.app.db.models.financial import (
    PayoutRecipient, Donation, PaymentOrder, PaymentCapture, Disbursement
)
from backend.app.services.disbursement_service import DisbursementService
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def maker_checker_setup(db_session):
    """Sets up eligible project and two distinct finance users."""
    now = datetime.now(timezone.utc)
    maker = User(
        id=str(uuid.uuid4()),
        email="operator1@charitychain.io",
        full_name="Finance Operator",
        hashed_password="mock",
        platform_role=PlatformRole.FINANCE_OPERATOR,
        is_verified=True
    )
    checker = User(
        id=str(uuid.uuid4()),
        email="approver1@charitychain.io",
        full_name="Finance Approver",
        hashed_password="mock",
        platform_role=PlatformRole.FINANCE_APPROVER,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Earth Initiative",
        legal_name="Clean Earth Initiative Inc",
        status=OrganizationStatus.VERIFIED
    )
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Reforestation",
        description="Tree planting in deforested watersheds.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=300000,
        raised_amount_minor_units=300000
    )
    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Seedling Nurseries",
        description="Seedling nursery establishment.",
        target_amount_minor_units=300000,
        raised_amount_minor_units=300000,
        status=MilestoneStatus.APPROVED
    )
    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        reconciliation_result=ReconciliationResult.APPROVED.value,
        snapshot_digest="valid-digest-mc",
        snapshot_data_json={"digest": "valid-digest-mc"}
    )
    recip = PayoutRecipient(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        recipient_type=RecipientType.PAYPAL_EMAIL,
        encrypted_destination="mock_enc",
        destination_masked="f***e@cleanearth.org",
        destination_hash="hash_mc",
        status=PayoutRecipientStatus.VERIFIED,
        is_primary=True
    )
    don = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=300000, status=DonationStatus.PAID)
    ord1 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don.id, provider_order_id="ORD-MC-1", amount_minor_units=300000, currency="USD", status=DonationStatus.PAID)
    cap = PaymentCapture(id=str(uuid.uuid4()), order_id=ord1.id, provider_capture_id="CAP-MC-1", amount_minor_units=300000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)

    db_session.add_all([maker, checker, org, project, milestone, review_case, recip, don, ord1, cap])
    db_session.commit()

    return maker, checker, milestone, recip


def test_self_approval_strictly_forbidden(db_session, maker_checker_setup):
    """Maker attempting to approve their own disbursement must be blocked."""
    maker, checker, milestone, recip = maker_checker_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    # 1. Maker requests disbursement
    disb = srv.request_disbursement(
        db=db_session,
        milestone_id=milestone.id,
        amount_minor_units=100000,
        requester_id=maker.id
    )
    assert disb.status == DisbursementStatus.PENDING_APPROVAL
    assert disb.requested_by_id == maker.id

    # 2. Maker attempts self-approval -> MUST RAISE ForbiddenException
    with pytest.raises(ForbiddenException) as exc:
        srv.approve_disbursement(
            db=db_session,
            disbursement_id=disb.id,
            approver_id=maker.id  # Same user!
        )
    assert "Maker-Checker financial control violation" in str(exc.value)


def test_independent_checker_approval_succeeds(db_session, maker_checker_setup):
    """Independent finance approver successfully authorizes disbursement."""
    maker, checker, milestone, recip = maker_checker_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    disb = srv.request_disbursement(
        db=db_session,
        milestone_id=milestone.id,
        amount_minor_units=100000,
        requester_id=maker.id
    )

    # Independent checker approves
    approved = srv.approve_disbursement(
        db=db_session,
        disbursement_id=disb.id,
        approver_id=checker.id,
        notes="Verified against signed milestone report"
    )

    assert approved.status == DisbursementStatus.APPROVED
    assert approved.approved_by_id == checker.id
    assert approved.approved_at is not None

