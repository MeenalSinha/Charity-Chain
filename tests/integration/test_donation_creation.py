"""
test_donation_creation.py - Donation Creation & Validation Integration Tests
=============================================================================
Tests donation creation rules, project and NGO eligibility checks,
currency match, and REST API endpoints (/api/v1/donations).
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus,
    ProjectVisibility, DonationStatus, PlatformRole
)
from backend.app.core.exceptions import ValidationException, CurrencyMismatchException
from backend.app.services.donation_service import DonationService
from tests.mocks.paypal_mock import MockPayPalProvider

client = TestClient(app)


@pytest.fixture
def active_project_setup(db_session):
    """Sets up a verified organization with an active public project and milestone."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Relief Foundation",
        legal_name="Global Relief Foundation Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Emergency Medical Shelter",
        description="Providing urgent triage shelter for disaster-affected families.",
        project_type="EMERGENCY_RELIEF",
        public_visibility=ProjectVisibility.PUBLIC,
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=500000,  # $5,000.00
        raised_amount_minor_units=0,
        is_archived=False
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Shelter Procurement",
        description="Procuring winterized shelter kits.",
        target_amount_minor_units=500000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add(milestone)
    db_session.commit()

    return org, project, milestone


def test_successful_donation_intent_creation(db_session, active_project_setup):
    """Verifies that creating a donation intent validates project/milestone and persists PaymentOrder."""
    org, project, milestone = active_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    res = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=2500,  # $25.00
        currency="USD",
        donor_name="Bob Contributor",
        donor_email="bob@example.com"
    )

    assert res["status"] == DonationStatus.PAYMENT_PENDING.value
    assert res["amount_minor_units"] == 2500
    assert res["currency"] == "USD"
    assert "MOCK-PAYPAL-ORD-" in res["provider_order_id"]
    assert "token=" in res["approval_url"]


def test_donation_intent_rejects_inactive_project(db_session, active_project_setup):
    """Verifies that donations to DRAFT or PAUSED projects are rejected."""
    org, project, milestone = active_project_setup
    project.status = ProjectStatus.DRAFT
    db_session.commit()

    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    with pytest.raises(ValidationException) as exc_info:
        srv.create_donation_intent(
            db=db_session,
            project_id=project.id,
            milestone_id=milestone.id,
            amount_minor_units=5000,
            currency="USD"
        )
    assert "must be active" in str(exc_info.value.message).lower()


def test_donation_intent_rejects_unverified_organization(db_session, active_project_setup):
    """Verifies that donations to projects of unverified NGOs are rejected."""
    org, project, milestone = active_project_setup
    org.status = OrganizationStatus.PENDING
    db_session.commit()

    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    with pytest.raises(ValidationException) as exc_info:
        srv.create_donation_intent(
            db=db_session,
            project_id=project.id,
            milestone_id=milestone.id,
            amount_minor_units=5000,
            currency="USD"
        )
    assert "not verified" in str(exc_info.value.message).lower()


def test_donation_intent_rejects_currency_mismatch(db_session, active_project_setup):
    """Verifies that currency must strictly match the project's currency."""
    org, project, milestone = active_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    with pytest.raises(CurrencyMismatchException):
        srv.create_donation_intent(
            db=db_session,
            project_id=project.id,
            milestone_id=milestone.id,
            amount_minor_units=5000,
            currency="EUR"  # Project is USD
        )


def test_donation_creation_rest_api(db_session, active_project_setup):
    """Verifies the REST API endpoint POST /api/v1/donations."""
    org, project, milestone = active_project_setup
    from backend.app.integrations.payments.factory import set_payment_provider
    from backend.app.api.v1.donations import create_donation
    from backend.app.api.schemas import CreateDonationRequest

    set_payment_provider(MockPayPalProvider())

    req = CreateDonationRequest(
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=5000,
        currency="USD",
        donor_name="Carol Donor",
        donor_email="carol@example.com",
        is_anonymous=False
    )

    response = create_donation(payload=req, current_user=None, db=db_session)
    assert response.success is True
    data = response.data
    assert data["status"] == "PAYMENT_PENDING"
    assert data["amount_minor_units"] == 5000
    assert "MOCK-PAYPAL-ORD-" in data["provider_order_id"]
