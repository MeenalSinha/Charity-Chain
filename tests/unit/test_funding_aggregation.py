"""
test_funding_aggregation.py - Funding Calculation & Integrity Unit Tests
========================================================================
Verifies that net funding accurately reflects settled captures adjusted for completed refunds.
Enforces integer minor units arithmetic without floating-point error.
"""

import pytest
import uuid
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture, PaymentRefund
from backend.app.core.constants import (
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus,
    PaymentRefundStatus, ProjectStatus, MilestoneStatus, OrganizationStatus
)
from backend.app.services.donation_service import DonationService


def test_project_and_milestone_funding_aggregation(db_session):
    """
    Verifies that ProjectService/DonationService computes:
    Net Funding = Gross Captures - Completed Refunds
    """
    # Provision Org, Project, and 2 Milestones
    org = Organization(
        id=str(uuid.uuid4()),
        name="Water For Life",
        legal_name="Water For Life Org",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Borehole Well Drilling",
        description="Clean water project for rural villages",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,  # $10,000.00
        raised_amount_minor_units=0
    )
    db_session.add(project)

    m1 = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Site Survey",
        description="Geological site survey",
        target_amount_minor_units=400000,  # $4,000.00
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    m2 = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=2,
        title="Drilling Operations",
        description="Borehole drilling operations",
        target_amount_minor_units=600000,  # $6,000.00
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add_all([m1, m2])
    db_session.commit()

    # Add 2 Donations:
    # Don 1 to Milestone 1: $1,000.00 (100,000 minor units)
    don1 = Donation(
        id=str(uuid.uuid4()),
        project_id=project.id,
        milestone_id=m1.id,
        amount_minor_units=100000,
        currency="USD",
        status=DonationStatus.PAID
    )
    ord1 = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=don1.id,
        amount_minor_units=100000,
        currency="USD",
        status=PaymentOrderStatus.COMPLETED
    )
    cap1 = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=ord1.id,
        provider_capture_id="CAP-111",
        amount_minor_units=100000,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED
    )

    # Don 2 to Milestone 2: $2,500.00 (250,000 minor units) with a partial refund of $500.00 (50,000 minor units)
    don2 = Donation(
        id=str(uuid.uuid4()),
        project_id=project.id,
        milestone_id=m2.id,
        amount_minor_units=250000,
        currency="USD",
        status=DonationStatus.PARTIALLY_REFUNDED
    )
    ord2 = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=don2.id,
        amount_minor_units=250000,
        currency="USD",
        status=PaymentOrderStatus.COMPLETED
    )
    cap2 = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=ord2.id,
        provider_capture_id="CAP-222",
        amount_minor_units=250000,
        currency="USD",
        status=PaymentCaptureStatus.PARTIALLY_REFUNDED
    )
    ref2 = PaymentRefund(
        id=str(uuid.uuid4()),
        capture_id=cap2.id,
        provider_refund_id="REF-222",
        amount_minor_units=50000,  # $500.00
        currency="USD",
        reason="Partial refund",
        status=PaymentRefundStatus.COMPLETED
    )

    db_session.add_all([don1, ord1, cap1, don2, ord2, cap2, ref2])
    db_session.commit()

    # Compute funding summary via DonationService
    srv = DonationService()
    summary = srv.get_project_funding_summary(db_session, project.id)

    # Expected:
    # Gross captures: 100,000 + 250,000 = 350,000 minor units ($3,500.00)
    # Total refunds: 50,000 minor units ($500.00)
    # Net funding: 300,000 minor units ($3,000.00)
    assert summary["gross_received_minor_units"] == 350000
    assert summary["refunded_minor_units"] == 50000
    assert summary["net_received_minor_units"] == 300000
    assert summary["percentage_funded"] == 30.0  # 3,000 / 10,000 * 100

    # Milestone breakdown
    ms_summaries = summary["milestones"]
    assert len(ms_summaries) == 2

    m1_sum = next(m for m in ms_summaries if m["milestone_id"] == m1.id)
    assert m1_sum["net_received_minor_units"] == 100000
    assert m1_sum["percentage_funded"] == 25.0  # 1,000 / 4,000 * 100

    m2_sum = next(m for m in ms_summaries if m["milestone_id"] == m2.id)
    assert m2_sum["gross_received_minor_units"] == 250000
    assert m2_sum["refunded_minor_units"] == 50000
    assert m2_sum["net_received_minor_units"] == 200000  # 250k - 50k
    assert round(m2_sum["percentage_funded"], 1) == 33.3  # 2,000 / 6,000 * 100
