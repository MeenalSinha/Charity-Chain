"""
test_public_transparency_api.py - Public Transparency API Integration Tests
===========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project, Milestone, ProjectVisibility
from backend.app.db.models.financial import PaymentCapture
from backend.app.core.constants import (
    PlatformRole, AccountStatus, ProjectStatus, MilestoneStatus,
    PaymentCaptureStatus, OrganizationStatus
)
from backend.app.services.analytics.transparency_service import PublicTransparencyService


@pytest.fixture
def transparency_test_env(db_session):
    user = User(
        id=str(uuid.uuid4()),
        email="public_creator@example.org",
        full_name="Creator User",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Reforestation Alliance",
        status=OrganizationStatus.VERIFIED,
        creator_id=user.id
    )
    # Public Project
    proj_public = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Amazon Rainforest Canopy Recovery",
        description="Publicly verified tree planting",
        public_visibility=ProjectVisibility.PUBLIC,
        status=ProjectStatus.ACTIVE,
        target_amount_minor_units=5000000,
        raised_amount_minor_units=2500000,
        currency="USD"
    )
    # Private / Internal Project (MUST NOT BE LEAKED TO PUBLIC LIST)
    proj_private = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Internal Facility Relocation",
        description="Confidential relocation plan",
        public_visibility=ProjectVisibility.PRIVATE,
        status=ProjectStatus.ACTIVE,
        target_amount_minor_units=100000,
        raised_amount_minor_units=10000,
        currency="USD"
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj_public.id,
        sequence_number=1,
        title="Nursery Planting",
        description="Plant 100,000 seedlings",
        target_amount_minor_units=2500000,
        currency="USD",
        status=MilestoneStatus.COMPLETED
    )
    c = PaymentCapture(
        id=str(uuid.uuid4()),
        amount_minor_units=2500000,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED
    )
    db_session.add_all([user, org, proj_public, proj_private, m, c])
    db_session.commit()
    return {"public_project": proj_public, "private_project": proj_private, "org": org}


def test_public_transparency_overview_and_project_listing(db_session, transparency_test_env):
    """Verifies that public transparency endpoints return only public projects and authoritative totals."""
    overview = PublicTransparencyService.get_global_overview(db_session)
    assert overview["organizations"]["verified"] >= 1
    assert overview["projects"]["published"] >= 1
    assert overview["financial_transparency"]["captured_cents"] >= 2500000

    # Project listing must ONLY return public projects
    projects = PublicTransparencyService.get_public_projects(db_session)
    titles = [p["title"] for p in projects]
    assert "Amazon Rainforest Canopy Recovery" in titles
    assert "Internal Facility Relocation" not in titles


def test_public_methodology_disclosure():
    """Verifies that public methodology provides clear calculation guidelines."""
    method = PublicTransparencyService.get_methodology()
    assert "methodology_version" in method
    assert "financial_calculation" in method
    assert "captured_donations" in method["financial_calculation"]
    assert "refunds" in method["financial_calculation"]
    assert "verification_and_review" in method
