"""
test_complete_disbursement_lifecycle.py - Complete Milestone Disbursement Lifecycle Functional Test
==================================================================================================
Comprehensive end-to-end functional test covering the entire Phase 8 financial architecture:
1. NGO Verification & Encrypted Payout Recipient Onboarding
2. Donor Contribution & Double-Entry Capture Posting
3. Evidence Submission & Dual-Key Human Review Approval
4. Multi-Gate Disbursement Eligibility Audit
5. Maker-Checker Segregation & Milestone Disbursement Authorization
6. Outbound Payout Batch Submission via PayPal
7. Double-Entry Accounting Ledger Invariant Verification (Debits == Credits)
8. Automated Financial Settlement Reconciliation
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta

from backend.app.core.constants import (
    PlatformRole, OrganizationStatus, ProjectStatus, MilestoneStatus,
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus,
    ReviewCaseStatus, ReviewDecision, ReconciliationResult,
    RecipientType, DisbursementMethod, PayoutRecipientStatus,
    DisbursementStatus, ReconciliationStatus, ReconciliationRunStatus
)
from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentCapture, PayoutRecipient, Disbursement,
    SettlementRecord
)
from backend.app.db.models.reviews_impact import ReviewCase
from backend.app.services.recipient_service import PayoutRecipientService
from backend.app.services.fund_allocation_service import FundAllocationService
from backend.app.services.ledger_service import LedgerService
from backend.app.services.disbursement_eligibility_service import DisbursementEligibilityService
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.reconciliation_service import ReconciliationService
from tests.mocks.payout_mock import MockPayoutProvider


def test_complete_milestone_disbursement_lifecycle(db_session):
    """
    Executes the complete CharityChain financial and disbursement lifecycle.
    """
    now = datetime.now(timezone.utc)
    mock_payout_provider = MockPayoutProvider()
    recip_service = PayoutRecipientService(provider=mock_payout_provider)
    disb_service = DisbursementService(provider=mock_payout_provider)

    # -------------------------------------------------------------------------
    # 1. Users Setup: Donor, Maker (Operator), Checker (Approver), Auditor
    # -------------------------------------------------------------------------
    donor = User(id=str(uuid.uuid4()), email="donor@globalgive.org", full_name="Philanthropist", hashed_password="mock", platform_role=PlatformRole.DONOR, is_verified=True)
    maker = User(id=str(uuid.uuid4()), email="maker.ops@charitychain.io", full_name="Finance Operator", hashed_password="mock", platform_role=PlatformRole.FINANCE_OPERATOR, is_verified=True)
    checker = User(id=str(uuid.uuid4()), email="checker.auth@charitychain.io", full_name="Finance Approver", hashed_password="mock", platform_role=PlatformRole.FINANCE_APPROVER, is_verified=True)
    auditor = User(id=str(uuid.uuid4()), email="auditor@charitychain.io", full_name="Finance Auditor", hashed_password="mock", platform_role=PlatformRole.FINANCE_AUDITOR, is_verified=True)
    db_session.add_all([donor, maker, checker, auditor])

    # -------------------------------------------------------------------------
    # 2. Organization & Verified Payout Destination
    # -------------------------------------------------------------------------
    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Reforestation Coalition",
        legal_name="Global Reforestation Coalition Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)
    db_session.commit()

    # Register NGO Payout Destination (PayPal email encrypted at rest)
    raw_payout_email = "disbursements@reforest.org"
    recip = recip_service.register_recipient(
        db=db_session,
        organization_id=org.id,
        destination=raw_payout_email,
        recipient_type=RecipientType.PAYPAL_EMAIL,
        is_primary=True,
        actor_id=maker.id
    )
    assert recip.status == PayoutRecipientStatus.PENDING_VERIFICATION

    # Verify recipient destination
    verified_recip = recip_service.verify_recipient(
        db=db_session,
        recipient_id=recip.id,
        verifier_id=maker.id,
        verification_notes="Verified against tax exempt certificate"
    )
    assert verified_recip.status == PayoutRecipientStatus.VERIFIED

    # -------------------------------------------------------------------------
    # 3. Project & Milestone Setup
    # -------------------------------------------------------------------------
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Amazon Corridor Restoration",
        description="Restoring 100 hectares of degraded rainforest.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,  # $10,000.00
        raised_amount_minor_units=0
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Sapling Procurement",
        description="Purchasing and staging 50,000 native tree saplings.",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.APPROVED
    )
    db_session.add(milestone)
    db_session.commit()

    # -------------------------------------------------------------------------
    # 4. Donor Contribution & Double-Entry Ledger Ingestion
    # -------------------------------------------------------------------------
    # Donor contributes $10,000.00 (1,000,000 cents)
    donation_amount = 1000000
    fee_amount = 29000  # $290.00 processing fee
    don = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=donation_amount,
        currency="USD",
        status=DonationStatus.PAID
    )
    ord_record = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=don.id,
        provider_order_id="PAYPAL-ORD-AMAZON-01",
        amount_minor_units=donation_amount,
        currency="USD",
        status=PaymentOrderStatus.COMPLETED
    )
    cap_record = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=ord_record.id,
        provider_capture_id="PAYPAL-CAP-AMAZON-01",
        amount_minor_units=donation_amount,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED,
        captured_at=now
    )
    db_session.add_all([don, ord_record, cap_record])
    db_session.commit()

    # Record double-entry donation capture
    capture_entry = LedgerService.record_donation_capture(
        db=db_session,
        donation=don,
        capture=cap_record,
        processor_fee_minor_units=fee_amount,
        actor_id=donor.id
    )
    assert capture_entry.is_reversed is False

    # Verify trial balance is in balance
    tb1 = LedgerService.get_trial_balance(db_session)
    assert tb1["is_balanced"] is True
    assert tb1["discrepancy_minor_units"] == 0

    # -------------------------------------------------------------------------
    # 5. Dual-Key Human Review Approval Setup
    # -------------------------------------------------------------------------
    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        reconciliation_result=ReconciliationResult.APPROVED.value,
        snapshot_digest="valid-crypto-digest-amazon-01",
        snapshot_data_json={"digest": "valid-crypto-digest-amazon-01"}
    )
    db_session.add(review_case)
    db_session.commit()

    # -------------------------------------------------------------------------
    # 6. Multi-Gate Eligibility Audit
    # -------------------------------------------------------------------------
    eligibility = DisbursementEligibilityService.evaluate_eligibility(
        db=db_session,
        milestone_id=milestone.id,
        requested_amount_minor_units=800000  # Requesting $8,000.00
    )
    assert eligibility["is_eligible"] is True
    assert eligibility["available_balance_minor_units"] == donation_amount
    assert len(eligibility["failed_checks"]) == 0

    # -------------------------------------------------------------------------
    # 7. Maker Step: Initiate Disbursement Request
    # -------------------------------------------------------------------------
    payout_amount = 800000  # $8,000.00
    disbursement = disb_service.request_disbursement(
        db=db_session,
        milestone_id=milestone.id,
        amount_minor_units=payout_amount,
        requester_id=maker.id,
        notes="Milestone sapling delivery verified by field drone inspection"
    )
    assert disbursement.status == DisbursementStatus.PENDING_APPROVAL
    assert disbursement.requested_by_id == maker.id

    # -------------------------------------------------------------------------
    # 8. Maker-Checker Segregation Enforcement
    # -------------------------------------------------------------------------
    # Maker attempts self-approval -> BLOCKED
    with pytest.raises(Exception):
        disb_service.approve_disbursement(
            db=db_session,
            disbursement_id=disbursement.id,
            approver_id=maker.id  # Same user
        )

    # Independent Checker approves -> SUCCESS
    approved_disb = disb_service.approve_disbursement(
        db=db_session,
        disbursement_id=disbursement.id,
        approver_id=checker.id,
        notes="Authorized pursuant to Dual-Key consensus"
    )
    assert approved_disb.status == DisbursementStatus.APPROVED
    assert approved_disb.approved_by_id == checker.id

    # Verify commitment ledger balance
    tb2 = LedgerService.get_trial_balance(db_session)
    assert tb2["is_balanced"] is True

    # -------------------------------------------------------------------------
    # 9. Outbound Batch Execution via PayPal
    # -------------------------------------------------------------------------
    batch = disb_service.create_and_submit_payout_batch(
        db=db_session,
        disbursement_ids=[approved_disb.id],
        submitter_id=checker.id,
        email_subject="Amazon Corridor Restoration — Milestone #1 Payout"
    )
    assert batch.status in (DisbursementStatus.SUCCEEDED, DisbursementStatus.PROCESSING)

    db_session.refresh(disbursement)
    assert disbursement.status == DisbursementStatus.SUCCEEDED
    assert disbursement.disbursed_at is not None
    assert disbursement.provider_item_id is not None

    # -------------------------------------------------------------------------
    # 10. Verify General Ledger Balance & Statements
    # -------------------------------------------------------------------------
    tb_final = LedgerService.get_trial_balance(db_session)
    assert tb_final["is_balanced"] is True
    assert tb_final["discrepancy_minor_units"] == 0

    # Remaining available funds on milestone should now be $2,000.00 (200,000 cents)
    summary_final = FundAllocationService.get_milestone_fund_summary(db_session, milestone.id)
    assert summary_final["net_raised_minor_units"] == 1000000
    assert summary_final["succeeded_payouts_minor_units"] == 800000
    assert summary_final["available_balance_minor_units"] == 200000

    # -------------------------------------------------------------------------
    # 11. Automated Financial Settlement Reconciliation
    # -------------------------------------------------------------------------
    # Ingest external settlement records matching capture and payout
    s_cap = SettlementRecord(
        id=str(uuid.uuid4()),
        transaction_id=cap_record.provider_capture_id,
        transaction_type="PAYMENT",
        gross_amount_minor_units=donation_amount,
        fee_amount_minor_units=fee_amount,
        net_amount_minor_units=donation_amount - fee_amount,
        currency="USD",
        settled_at=now
    )
    s_payout = SettlementRecord(
        id=str(uuid.uuid4()),
        transaction_id=disbursement.provider_item_id,
        transaction_type="PAYOUT",
        gross_amount_minor_units=payout_amount,
        fee_amount_minor_units=25,
        net_amount_minor_units=payout_amount,
        currency="USD",
        settled_at=now
    )
    db_session.add_all([s_cap, s_payout])
    db_session.commit()

    rec_run = ReconciliationService.run_reconciliation(
        db=db_session,
        period_start=now - timedelta(days=1),
        period_end=now + timedelta(days=1),
        executed_by_id=auditor.id
    )

    assert rec_run.status == ReconciliationRunStatus.COMPLETED
    assert rec_run.matched_count == 2
    assert rec_run.discrepancy_count == 0

