"""
test_project_discovery.py - Integration tests for Public Project Discovery & Filtering
=====================================================================================
Tests public visibility filters, search queries, category filters, and pagination.
"""

import pytest
from backend.app.core.constants import (
    OrganizationType, OrganizationStatus, PlatformRole, OrganizationRole,
    ProjectStatus, ProjectVisibility
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, CreateProjectRequest, CreateMilestoneRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import RegisterRequest
from backend.app.db.models.users import User, AccountStatus, Organization
from backend.app.db.models.projects import Project


@pytest.fixture
def discovery_data(db_session):
    """Set up verified and unverified organizations, active and draft projects."""
    # User
    reg = AuthService.register_user(db_session, RegisterRequest(
        full_name="Discovery Lead",
        email="discovery@ngo.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u = db_session.get(User, reg.user_id)
    u.is_verified = True
    u.account_status = AccountStatus.ACTIVE

    ProjectService.seed_default_categories(db_session)
    cats = ProjectService.list_categories(db_session)
    env_cat = next(c for c in cats if "environ" in c.slug)
    edu_cat = next(c for c in cats if c.slug == "education")

    # 1. Verified Org
    org1 = NGOOnboardingService.create_organization(db_session, CreateOrganizationRequest(
        legal_name="Verified Wildlife Trust",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="Gov Authority",
        registration_number="REG-001",
        country_of_registration="US",
        registered_address="Denver, CO",
        operating_address="Denver, CO",
        primary_operating_region="Rocky Mountains",
        mission_statement="Protecting apex predators."
    ), reg.user_id)
    db_org1 = db_session.get(Organization, org1.id)
    db_org1.status = OrganizationStatus.VERIFIED

    # 2. Unverified (Draft) Org
    org2 = NGOOnboardingService.create_organization(db_session, CreateOrganizationRequest(
        legal_name="Unverified Local Group",
        organization_type=OrganizationType.NGO,
        registration_authority="Gov Authority",
        registration_number="REG-002",
        country_of_registration="US",
        registered_address="Seattle, WA",
        operating_address="Seattle, WA",
        primary_operating_region="Pacific Northwest",
        mission_statement="Local cleanup."
    ), reg.user_id)

    # 3. Active Project from Verified Org in Environment
    p1 = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org1.id,
        category_id=env_cat.id,
        title="Grizzly Bear Habitat Restoration",
        summary="Restoring 100 acres of degraded grizzly feeding grounds.",
        objectives="Plant berry bushes and decommission logging roads.",
        funding_target_minor=100000,
        currency="USD",
        country="US",
        primary_operating_region="Rocky Mountains"
    ), reg.user_id)
    db_p1 = db_session.get(Project, p1.id)
    db_p1.status = ProjectStatus.ACTIVE

    # 4. Active Project from Verified Org in Education
    p2 = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org1.id,
        category_id=edu_cat.id,
        title="Rural STEM Education Program",
        summary="Supplying laptops and internet access to 10 rural schools.",
        objectives="Enhance digital literacy for underserved students.",
        funding_target_minor=200000,
        currency="USD",
        country="US",
        primary_operating_region="Appalachia"
    ), reg.user_id)
    db_p2 = db_session.get(Project, p2.id)
    db_p2.status = ProjectStatus.ACTIVE

    # 5. Draft Project from Verified Org (Should NOT be discovered)
    p3 = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org1.id,
        category_id=env_cat.id,
        title="Wolf Sanctuary Expansion",
        summary="Planning fence line expansion for wolf reserve.",
        objectives="Build 5km predator-proof perimeter fence.",
        funding_target_minor=300000,
        currency="USD",
        country="US",
        primary_operating_region="Rocky Mountains"
    ), reg.user_id)

    # 6. Active Project from Unverified Org (Should NOT be discovered)
    p4 = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org2.id,
        category_id=env_cat.id,
        title="Unverified Urban Tree Project",
        summary="Tree planting initiative.",
        objectives="Plant street trees.",
        funding_target_minor=50000,
        currency="USD",
        country="US",
        primary_operating_region="Pacific Northwest"
    ), reg.user_id)
    db_p4 = db_session.get(Project, p4.id)
    db_p4.status = ProjectStatus.ACTIVE

    db_session.commit()
    return {
        "env_cat_id": env_cat.id,
        "edu_cat_id": edu_cat.id,
        "active_p1_id": p1.id,
        "active_p2_id": p2.id,
        "draft_p3_id": p3.id,
        "unverified_org_p4_id": p4.id
    }


def test_public_discovery_filters_strictly(db_session, discovery_data):
    """Verify only ACTIVE projects from VERIFIED orgs appear in public discovery."""
    results = ProjectService.list_public_projects(db_session, page=1, page_size=10)
    
    # Must only return Grizzly Bear Habitat and Rural STEM
    returned_ids = [p.id for p in results.items]
    assert discovery_data["active_p1_id"] in returned_ids
    assert discovery_data["active_p2_id"] in returned_ids
    assert discovery_data["draft_p3_id"] not in returned_ids
    assert discovery_data["unverified_org_p4_id"] not in returned_ids
    assert results.total_count == 2


def test_public_discovery_keyword_search(db_session, discovery_data):
    """Verify keyword search matches project title and summary."""
    # Search for "Grizzly"
    res1 = ProjectService.list_public_projects(db_session, search="Grizzly")
    assert res1.total_count == 1
    assert res1.items[0].id == discovery_data["active_p1_id"]

    # Search for "Laptops" (matches summary of STEM project)
    res2 = ProjectService.list_public_projects(db_session, search="laptops")
    assert res2.total_count == 1
    assert res2.items[0].id == discovery_data["active_p2_id"]

    # Non-matching search
    res3 = ProjectService.list_public_projects(db_session, search="NonExistentKeywordXYZ")
    assert res3.total_count == 0


def test_public_discovery_category_filter(db_session, discovery_data):
    """Verify filtering by category."""
    res_env = ProjectService.list_public_projects(db_session, category_id=discovery_data["env_cat_id"])
    assert res_env.total_count == 1
    assert res_env.items[0].id == discovery_data["active_p1_id"]

    res_edu = ProjectService.list_public_projects(db_session, category_id=discovery_data["edu_cat_id"])
    assert res_edu.total_count == 1
    assert res_edu.items[0].id == discovery_data["active_p2_id"]
