"""
test_ngo_analytics_api.py - Integration Tests for NGO Analytics & Reports
========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import Donation, Disbursement
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, AccountStatus, ProjectStatus,
    MilestoneStatus, DonationStatus, DisbursementStatus
)
from backend.app.services.analytics.ngo_analytics_service import NGOAnalyticsService


@pytest.fixture
def test_ngo_env(db_session):
    user = User(
        id=str(uuid.uuid4()),
        email=f"ngo_admin_{uuid.uuid4().hex[:6]}@example.org",
        full_name="NGO Admin Test",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Wildlife Conservation Org",
        creator_id=user.id
    )
    membership = OrganizationMembership(
        id=str(uuid.uuid4()),
        user_id=user.id,
        organization_id=org.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Rhino Habitat Protection",
        description="Preserving savanna sanctuaries",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=600000,
        currency="USD",
        status=ProjectStatus.ACTIVE
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Sanctuary Fencing",
        description="Install perimeter protection",
        target_amount_minor_units=600000,
        currency="USD",
        status=MilestoneStatus.COMPLETED
    )
    d = Donation(
        id=str(uuid.uuid4()),
        donor_id=user.id,
        project_id=proj.id,
        milestone_id=m.id,
        amount_minor_units=600000,
        currency="USD",
        status=DonationStatus.PAID
    )
    disb = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id=m.id,
        organization_id=org.id,
        amount_minor_units=500000,
        currency="USD",
        status=DisbursementStatus.SUCCEEDED
    )
    db_session.add_all([user, org, membership, proj, m, d, disb])
    db_session.commit()
    return {"user": user, "org": org, "proj": proj, "milestone": m}


def test_ngo_overview_service(db_session, test_ngo_env):
    """Verifies that NGO overview accurately reflects projects, funding, and available balances."""
    org_id = test_ngo_env["org"].id
    overview = NGOAnalyticsService.get_ngo_overview(db_session, org_id)

    assert overview["organization_id"] == org_id
    assert overview["projects"]["total"] == 1
    assert overview["projects"]["active"] == 1
    assert overview["milestones"]["completed"] == 1

    fin = overview["financial"]
    assert fin["total_captured_cents"] == 600000
    assert fin["confirmed_disbursements_cents"] == 500000
    assert fin["available_funds_cents"] == 100000
    assert fin["available_funds_formatted"] == "$1,000.00"


def test_ngo_impact_report_generation(db_session, test_ngo_env):
    """Verifies that the generated impact report includes all required sections and disclaimer."""
    org_id = test_ngo_env["org"].id
    report = NGOAnalyticsService.generate_impact_report(db_session, org_id)

    assert "report_id" in report
    assert report["organization_name"] == "Wildlife Conservation Org"
    assert "overview" in report
    assert "funding_summary" in report
    assert "verified_impact" in report
    assert "disclaimer" in report
    assert "CharityChain ledger" in report["disclaimer"]
