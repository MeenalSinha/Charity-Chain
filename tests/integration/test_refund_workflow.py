"""
test_refund_workflow.py - Refund Processing & Net Funding Adjustment Tests
===========================================================================
Tests full and partial refund processing, balance ceilings, and funding adjustments.
"""

import pytest
import uuid
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus, DonationStatus
)
from backend.app.services.donation_service import DonationService
from backend.app.integrations.payments.exceptions import PaymentRefundFailedException
from tests.mocks.paypal_mock import MockPayPalProvider


@pytest.fixture
def refund_project_setup(db_session):
    """Sets up project for refund testing."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Amazon Forest Protection",
        legal_name="Amazon Forest Protection Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Reforestation Initiative",
        description="Planting native tree seedlings in degraded riparian corridors.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Seedling Nurseries",
        description="Establishing shade-grown seedling nurseries.",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add(milestone)
    db_session.commit()

    return org, project, milestone


def test_partial_and_full_refund_workflow(db_session, refund_project_setup):
    """Verifies that multiple partial refunds reduce balance and adjust net funding totals."""
    org, project, milestone = refund_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    # 1. Create and capture a $100.00 donation (10,000 minor units)
    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=10000,
        currency="USD",
        donor_id="donor-123"
    )
    srv.capture_donation_payment(db=db_session, order_identifier=intent["provider_order_id"])

    db_session.refresh(project)
    assert project.raised_amount_minor_units == 10000

    # 2. First partial refund: $40.00 (4,000 minor units)
    ref1 = srv.process_refund(
        db=db_session,
        donation_id=intent["donation_id"],
        amount_minor_units=4000,
        reason="Donor requested partial refund",
        actor_id="donor-123"
    )

    assert ref1["status"] == "COMPLETED"
    assert ref1["donation_status"] == DonationStatus.PARTIALLY_REFUNDED.value
    assert ref1["remaining_refundable_minor_units"] == 6000

    db_session.refresh(project)
    db_session.refresh(milestone)
    assert project.raised_amount_minor_units == 6000
    assert milestone.raised_amount_minor_units == 6000

    # 3. Attempting to refund more than remaining ($70.00 > $60.00) must fail
    with pytest.raises(PaymentRefundFailedException):
        srv.process_refund(
            db=db_session,
            donation_id=intent["donation_id"],
            amount_minor_units=7000,
            reason="Exceeding remaining balance",
            actor_id="donor-123"
        )

    # 4. Final remaining refund: $60.00 (6,000 minor units)
    ref2 = srv.process_refund(
        db=db_session,
        donation_id=intent["donation_id"],
        amount_minor_units=6000,
        reason="Final refund",
        actor_id="donor-123"
    )

    assert ref2["status"] == "COMPLETED"
    assert ref2["donation_status"] == DonationStatus.REFUNDED.value
    assert ref2["remaining_refundable_minor_units"] == 0

    db_session.refresh(project)
    db_session.refresh(milestone)
    assert project.raised_amount_minor_units == 0
    assert milestone.raised_amount_minor_units == 0
