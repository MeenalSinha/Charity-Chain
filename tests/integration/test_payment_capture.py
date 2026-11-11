"""
test_payment_capture.py - Payment Capture & Idempotency Integration Tests
==========================================================================
Tests checkout capture, gateway status verification, capture record persistence,
idempotency, and funding aggregate updates.
"""

import pytest
import uuid
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus,
    ProjectVisibility, DonationStatus, PaymentOrderStatus
)
from backend.app.services.donation_service import DonationService
from tests.mocks.paypal_mock import MockPayPalProvider


@pytest.fixture
def capture_project_setup(db_session):
    """Sets up a verified organization with an active project."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Solar Light Foundation",
        legal_name="Solar Light Foundation Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Rural Solar Installation",
        description="Installing off-grid solar panels in community clinics.",
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
        title="Solar Panel Delivery",
        description="Delivery and staging of solar array hardware.",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add(milestone)
    db_session.commit()

    return org, project, milestone


def test_successful_payment_capture_and_funding_update(db_session, capture_project_setup):
    """Verifies that capturing an approved order marks donation PAID and increments raised funds."""
    org, project, milestone = capture_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    # 1. Create donation intent
    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=15000,  # $150.00
        currency="USD"
    )

    # 2. Capture payment
    capture_res = srv.capture_donation_payment(
        db=db_session,
        order_identifier=intent["provider_order_id"]
    )

    assert capture_res["status"] == "COMPLETED"
    assert capture_res["donation_status"] == DonationStatus.PAID.value
    assert "MOCK-CAP-" in capture_res["provider_capture_id"]

    # 3. Verify funding total updated on milestone and project
    db_session.refresh(milestone)
    db_session.refresh(project)
    assert milestone.raised_amount_minor_units == 15000
    assert project.raised_amount_minor_units == 15000


def test_capture_idempotency_returns_same_result(db_session, capture_project_setup):
    """Verifies that calling capture multiple times does not capture twice or double-count funds."""
    org, project, milestone = capture_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=10000,
        currency="USD"
    )

    # First capture
    res1 = srv.capture_donation_payment(db=db_session, order_identifier=intent["provider_order_id"])

    # Second capture (same order)
    res2 = srv.capture_donation_payment(db=db_session, order_identifier=intent["provider_order_id"])

    assert res1["capture_id"] == res2["capture_id"]
    assert res1["provider_capture_id"] == res2["provider_capture_id"]

    # Funding must not double-count
    db_session.refresh(milestone)
    db_session.refresh(project)
    assert milestone.raised_amount_minor_units == 10000
    assert project.raised_amount_minor_units == 10000
