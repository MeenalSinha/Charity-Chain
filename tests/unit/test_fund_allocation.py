"""
test_fund_allocation.py - Real-Time Fund Balance & Allocation Unit Tests
========================================================================
Tests exact calculation of gross captures, refunds, commitments, payouts,
and unencumbered balances in integer minor units.
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus,
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus,
    PaymentRefundStatus, DisbursementStatus
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentCapture, PaymentRefund, Disbursement
)
from backend.app.services.fund_allocation_service import FundAllocationService


@pytest.fixture
def allocation_setup(db_session):
    """Sets up an org, project, and milestone with initial financial records."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean School Foundation",
        legal_name="Clean School Foundation Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Sanitation Facilities",
        description="Building clean sanitation blocks in elementary schools.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=100000,  # $1,000.00
        raised_amount_minor_units=0
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Plumbing Stage",
        description="Water supply lines and septic tanks.",
        target_amount_minor_units=100000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.APPROVED
    )
    db_session.add(milestone)
    db_session.commit()

    return org, project, milestone


def test_fund_allocation_balance_calculations(db_session, allocation_setup):
    """Verifies calculations across captures, refunds, commitments, and payouts."""
    org, project, milestone = allocation_setup
    now = datetime.now(timezone.utc)

    # 1. Add Donation #1 ($600.00 / 60,000 cents) captured
    don1 = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=60000, status=DonationStatus.PAID)
    ord1 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don1.id, provider_order_id="ORD-1", amount_minor_units=60000, currency="USD", status=PaymentOrderStatus.COMPLETED)
    cap1 = PaymentCapture(id=str(uuid.uuid4()), order_id=ord1.id, provider_capture_id="CAP-1", amount_minor_units=60000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)
    db_session.add_all([don1, ord1, cap1])

    # 2. Add Donation #2 ($400.00 / 40,000 cents) captured
    don2 = Donation(id=str(uuid.uuid4()), project_id=project.id, milestone_id=milestone.id, amount_minor_units=40000, status=DonationStatus.PAID)
    ord2 = PaymentOrder(id=str(uuid.uuid4()), donation_id=don2.id, provider_order_id="ORD-2", amount_minor_units=40000, currency="USD", status=PaymentOrderStatus.COMPLETED)
    cap2 = PaymentCapture(id=str(uuid.uuid4()), order_id=ord2.id, provider_capture_id="CAP-2", amount_minor_units=40000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)
    db_session.add_all([don2, ord2, cap2])
    db_session.commit()

    summary1 = FundAllocationService.get_milestone_fund_summary(db_session, milestone.id)
    assert summary1["gross_captured_minor_units"] == 100000  # $1,000.00
    assert summary1["completed_refunds_minor_units"] == 0
    assert summary1["net_raised_minor_units"] == 100000
    assert summary1["available_balance_minor_units"] == 100000
    assert summary1["funding_percentage"] == 100.0

    # 3. Add a partial refund of $150.00 (15,000 cents) on cap2
    ref1 = PaymentRefund(
        id=str(uuid.uuid4()),
        capture_id=cap2.id,
        provider_refund_id="REF-1",
        amount_minor_units=15000,
        currency="USD",
        status=PaymentRefundStatus.COMPLETED,
        refunded_at=now
    )
    db_session.add(ref1)
    db_session.commit()

    summary2 = FundAllocationService.get_milestone_fund_summary(db_session, milestone.id)
    assert summary2["gross_captured_minor_units"] == 100000
    assert summary2["completed_refunds_minor_units"] == 15000
    assert summary2["net_raised_minor_units"] == 85000  # $850.00
    assert summary2["available_balance_minor_units"] == 85000

    # 4. Add an approved payout commitment of $300.00 (30,000 cents)
    disb1 = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        organization_id=org.id,
        amount_minor_units=30000,
        currency="USD",
        status=DisbursementStatus.APPROVED
    )
    db_session.add(disb1)
    db_session.commit()

    summary3 = FundAllocationService.get_milestone_fund_summary(db_session, milestone.id)
    assert summary3["committed_payouts_minor_units"] == 30000
    assert summary3["succeeded_payouts_minor_units"] == 0
    assert summary3["available_balance_minor_units"] == 55000  # 85,000 - 30,000 = 55,000

    # 5. Settle disbursement #1
    disb1.status = DisbursementStatus.SUCCEEDED
    disb1.disbursed_at = now
    db_session.commit()

    summary4 = FundAllocationService.get_milestone_fund_summary(db_session, milestone.id)
    assert summary4["committed_payouts_minor_units"] == 0
    assert summary4["succeeded_payouts_minor_units"] == 30000
    assert summary4["available_balance_minor_units"] == 55000

