"""
test_organization_isolation.py - Security tests for Organization Multi-Tenant Isolation
=======================================================================================
Verifies that users of Organization A cannot access, mutate, or manage
Organization B's resources, documents, projects, or milestones (IDOR defense).
"""

import pytest
from backend.app.core.constants import OrganizationType, PlatformRole, OrganizationRole
from backend.app.api.schemas import (
    CreateOrganizationRequest, UpdateOrganizationRequest,
    CreateProjectRequest, CreateMilestoneRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import RegisterRequest
from backend.app.core.exceptions import ForbiddenException
from backend.app.db.models.users import User, AccountStatus


@pytest.fixture
def multi_org_setup(db_session):
    """Set up two separate users each owning a separate organization."""
    # User A & Org A
    reg_a = AuthService.register_user(db_session, RegisterRequest(
        full_name="User Alpha",
        email="alpha@orga.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    ua = db_session.get(User, reg_a.user_id)
    ua.is_verified = True
    ua.account_status = AccountStatus.ACTIVE

    org_a = NGOOnboardingService.create_organization(db_session, CreateOrganizationRequest(
        legal_name="Organization Alpha",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="Auth A",
        registration_number="REG-A",
        country_of_registration="US",
        registered_address="1 Alpha Way",
        operating_address="1 Alpha Way",
        primary_operating_region="East",
        mission_statement="Mission Alpha"
    ), reg_a.user_id)

    # User B & Org B
    reg_b = AuthService.register_user(db_session, RegisterRequest(
        full_name="User Beta",
        email="beta@orgb.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    ub = db_session.get(User, reg_b.user_id)
    ub.is_verified = True
    ub.account_status = AccountStatus.ACTIVE

    org_b = NGOOnboardingService.create_organization(db_session, CreateOrganizationRequest(
        legal_name="Organization Beta",
        organization_type=OrganizationType.CHARITY,
        registration_authority="Auth B",
        registration_number="REG-B",
        country_of_registration="GB",
        registered_address="2 Beta Blvd",
        operating_address="2 Beta Blvd",
        primary_operating_region="West",
        mission_statement="Mission Beta"
    ), reg_b.user_id)

    # Org B Project
    ProjectService.seed_default_categories(db_session)
    cats = ProjectService.list_categories(db_session)
    proj_b = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org_b.id,
        category_id=cats[0].id,
        title="Project Beta One",
        summary="Confidential summary of Project Beta One.",
        objectives="Project Beta Objectives.",
        funding_target_minor=100000,
        currency="USD",
        country="GB",
        primary_operating_region="West"
    ), reg_b.user_id)

    db_session.commit()
    return {
        "user_a": reg_a.user_id,
        "org_a": org_a.id,
        "user_b": reg_b.user_id,
        "org_b": org_b.id,
        "proj_b": proj_b.id
    }


def test_cross_org_profile_access_denied(db_session, multi_org_setup):
    """Verify User A cannot access details of Org B."""
    user_a = multi_org_setup["user_a"]
    org_b = multi_org_setup["org_b"]

    with pytest.raises(ForbiddenException):
        NGOOnboardingService.get_organization_details(db_session, org_b, user_a)


def test_cross_org_document_listing_denied(db_session, multi_org_setup):
    """Verify User A cannot list verification documents of Org B."""
    user_a = multi_org_setup["user_a"]
    org_b = multi_org_setup["org_b"]

    with pytest.raises(ForbiddenException):
        NGOOnboardingService.list_organization_documents(db_session, org_b, user_a)


def test_cross_org_project_creation_denied(db_session, multi_org_setup):
    """Verify User A cannot create a project under Org B."""
    user_a = multi_org_setup["user_a"]
    org_b = multi_org_setup["org_b"]
    cats = ProjectService.list_categories(db_session)

    with pytest.raises(ForbiddenException):
        ProjectService.create_project(db_session, CreateProjectRequest(
            organization_id=org_b,  # Target Org B
            category_id=cats[0].id,
            title="Unauthorized Project Injection",
            summary="Attempting to inject project into Org B.",
            objectives="Test objectives.",
            funding_target_minor=50000,
            currency="USD",
            country="GB",
            primary_operating_region="West"
        ), user_a)  # User A


def test_cross_org_milestone_injection_denied(db_session, multi_org_setup):
    """Verify User A cannot attach milestones to Org B's project."""
    user_a = multi_org_setup["user_a"]
    proj_b = multi_org_setup["proj_b"]

    m_req = CreateMilestoneRequest(
        sequence_number=1,
        title="Unauthorized Milestone",
        objective="Inject milestone into another orgs project",
        funding_target_minor=10000,
        currency="USD"
    )
    with pytest.raises(ForbiddenException):
        ProjectService.create_milestone(db_session, proj_b, m_req, user_a)

