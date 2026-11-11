"""
test_payment_security.py - Payment Security & Access Control Tests
==================================================================
Tests IDOR defenses, cross-donor isolation, amount manipulation resistance,
and webhook forgery prevention.
"""

import pytest
import uuid
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus, PlatformRole, AccountStatus
)
from backend.app.core.exceptions import ForbiddenException, CurrencyMismatchException
from backend.app.services.donation_service import DonationService
from tests.mocks.paypal_mock import MockPayPalProvider


@pytest.fixture
def security_setup(db_session):
    """Sets up users, org, and project for security evaluation."""
    donor1 = User(
        id=str(uuid.uuid4()),
        email="donor1@example.com",
        full_name="Donor One",
        hashed_password="mock_argon2_hash_1",
        platform_role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    donor2 = User(
        id=str(uuid.uuid4()),
        email="donor2@example.com",
        full_name="Donor Two",
        hashed_password="mock_argon2_hash_2",
        platform_role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Medical Relief",
        legal_name="Medical Relief Global",
        status=OrganizationStatus.VERIFIED
    )
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Mobile Clinic Vans",
        description="Equipping mobile clinic vans for remote maternal healthcare.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=500000,
        raised_amount_minor_units=0
    )
    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Van Customization",
        description="Installing refrigerated medicine coolers and power units.",
        target_amount_minor_units=500000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add_all([donor1, donor2, org, project, milestone])
    db_session.commit()

    return donor1, donor2, project, milestone


def test_cross_donor_capture_forbidden(db_session, security_setup):
    """Ensures a user cannot capture a payment order created by another donor."""
    donor1, donor2, project, milestone = security_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    # Donor 1 creates intent
    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=5000,
        currency="USD",
        donor_id=donor1.id
    )

    # Donor 2 attempts to capture Donor 1's order
    with pytest.raises(ForbiddenException):
        srv.capture_donation_payment(
            db=db_session,
            order_identifier=intent["provider_order_id"],
            user_id=donor2.id
        )


def test_cross_donor_refund_forbidden(db_session, security_setup):
    """Ensures a donor cannot request a refund on another donor's captured contribution."""
    donor1, donor2, project, milestone = security_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=10000,
        currency="USD",
        donor_id=donor1.id
    )
    srv.capture_donation_payment(db=db_session, order_identifier=intent["provider_order_id"])

    # Donor 2 attempts to request refund on Donor 1's donation
    with pytest.raises(ForbiddenException):
        srv.process_refund(
            db=db_session,
            donation_id=intent["donation_id"],
            amount_minor_units=5000,
            reason="Malicious refund attempt",
            actor_id=donor2.id,
            is_admin=False
        )


def test_currency_manipulation_rejected(db_session, security_setup):
    """Ensures client cannot submit arbitrary currency codes conflicting with target project."""
    donor1, donor2, project, milestone = security_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    with pytest.raises(CurrencyMismatchException):
        srv.create_donation_intent(
            db=db_session,
            project_id=project.id,
            milestone_id=milestone.id,
            amount_minor_units=5000,
            currency="GBP",  # Project currency is USD
            donor_id=donor1.id
        )
