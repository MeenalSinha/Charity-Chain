"""
test_project_crud.py - Integration tests for Project & Milestone CRUD and Validation
===================================================================================
Tests category seeding, project creation, milestone allocations, pre-submission
validation report evaluation, and project submission.
"""

import pytest
from backend.app.core.constants import (
    OrganizationType, OrganizationStatus, PlatformRole, OrganizationRole,
    ProjectStatus, ProjectVisibility
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, CreateProjectRequest,
    CreateMilestoneRequest, UpdateMilestoneRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import RegisterRequest
from backend.app.core.exceptions import ValidationException, ForbiddenException
from backend.app.db.models.users import User, AccountStatus, Organization


@pytest.fixture
def project_setup(db_session):
    """Set up an active user, a verified organization, and seeded categories."""
    # User
    reg = AuthService.register_user(db_session, RegisterRequest(
        full_name="Maria Project Lead",
        email="maria.lead@conservation.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u = db_session.get(User, reg.user_id)
    u.is_verified = True
    u.account_status = AccountStatus.ACTIVE

    # Organization
    org_req = CreateOrganizationRequest(
        legal_name="Amazon Conservation Guild",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="Brazilian Environment Agency",
        registration_number="IBAMA-998811",
        country_of_registration="BR",
        registered_address="Manaus, AM",
        operating_address="Manaus, AM",
        primary_operating_region="Amazonas",
        mission_statement="Protecting biodiversity hotspots."
    )
    org = NGOOnboardingService.create_organization(db_session, org_req, reg.user_id)
    
    # Mark org verified for project submission tests
    db_org = db_session.get(Organization, org.id)
    db_org.status = OrganizationStatus.VERIFIED
    
    # Seed default categories
    ProjectService.seed_default_categories(db_session)
    cats = ProjectService.list_categories(db_session)
    
    db_session.commit()
    return {
        "user_id": reg.user_id,
        "org_id": org.id,
        "category_id": cats[0].id
    }


def test_project_creation_and_milestone_budget_enforcement(db_session, project_setup):
    """Verify project creation, milestone addition, and target budget bounds."""
    user_id = project_setup["user_id"]
    org_id = project_setup["org_id"]
    cat_id = project_setup["category_id"]

    # 1. Create Project ($10,000 USD = 1,000,000 minor units)
    p_req = CreateProjectRequest(
        organization_id=org_id,
        category_id=cat_id,
        title="Reforesting Deforested Corridors",
        summary="Planting 25,000 native tree saplings across corridor corridors.",
        objectives="Restore native tree canopy and protect endangered tamarin monkeys.",
        project_type="CONSERVATION",
        public_visibility=ProjectVisibility.PUBLIC,
        funding_target_minor=1000000,
        currency="USD",
        country="BR",
        primary_operating_region="Amazonas",
        beneficiary_location="Manaus Basin",
        expected_beneficiaries=12000
    )
    project = ProjectService.create_project(db_session, p_req, user_id)
    assert project.id is not None
    assert project.status == ProjectStatus.DRAFT.value
    assert project.funding_target_minor == 1000000

    # 2. Add Milestone 1: $6,000 USD (600,000 minor units)
    m1_req = CreateMilestoneRequest(
        sequence_number=1,
        title="Sapling Nursery Setup",
        objective="Construct nursery and germinate 25,000 native tree saplings",
        funding_target_minor=600000,
        currency="USD",
        target_latitude=-3.119028,
        target_longitude=-60.021731
    )
    m1 = ProjectService.create_milestone(db_session, project.id, m1_req, user_id)
    assert m1.sequence_number == 1

    # 3. Add Milestone 2 exceeding target ($5,000 USD -> 600k + 500k = 1.1M > 1.0M)
    m2_over_req = CreateMilestoneRequest(
        sequence_number=2,
        title="Outplanting Saplings",
        objective="Plant trees in primary corridor",
        funding_target_minor=500000,  # Total would exceed 1,000,000
        currency="USD",
        target_latitude=-3.120000,
        target_longitude=-60.025000
    )
    with pytest.raises(ValidationException) as exc_info:
        ProjectService.create_milestone(db_session, project.id, m2_over_req, user_id)
    assert "exceed" in str(exc_info.value.message).lower() and "target" in str(exc_info.value.message).lower()

    # 4. Add Milestone 2 with exact remaining balance ($4,000 USD = 400,000 minor units)
    m2_exact_req = CreateMilestoneRequest(
        sequence_number=2,
        title="Outplanting Saplings",
        objective="Plant trees in primary corridor",
        funding_target_minor=400000,
        currency="USD",
        target_latitude=-3.120000,
        target_longitude=-60.025000
    )
    m2 = ProjectService.create_milestone(db_session, project.id, m2_exact_req, user_id)
    assert m2.sequence_number == 2


def test_project_pre_submission_validation_and_submission(db_session, project_setup):
    """Verify 13-rule validation checklist and submission flow."""
    user_id = project_setup["user_id"]
    org_id = project_setup["org_id"]
    cat_id = project_setup["category_id"]

    # Create project
    p_req = CreateProjectRequest(
        organization_id=org_id,
        category_id=cat_id,
        title="River Cleanliness Monitoring",
        summary="Installing continuous water monitoring telemetry stations.",
        objectives="Real-time water quality monitoring for community safety.",
        project_type="ENVIRONMENTAL",
        funding_target_minor=500000,
        currency="USD",
        country="BR",
        primary_operating_region="Amazonas"
    )
    project = ProjectService.create_project(db_session, p_req, user_id)

    # Incomplete project (no milestones) validation report
    val1 = ProjectService.validate_project_submission(db_session, project.id, user_id)
    assert not val1.is_valid
    assert any(i.code in ("NO_MILESTONES", "MILESTONES_REQUIRED") for i in val1.issues)

    # Submitting incomplete project fails
    with pytest.raises(ValidationException):
        ProjectService.submit_project(db_session, project.id, user_id)

    # Add complete milestone matching full budget
    m_req = CreateMilestoneRequest(
        sequence_number=1,
        title="Install Telemetry Nodes",
        objective="Deploy 10 solar telemetry sensor nodes",
        funding_target_minor=500000,
        currency="USD"
    )
    ProjectService.create_milestone(db_session, project.id, m_req, user_id)

    # Validation now passes completely
    val2 = ProjectService.validate_project_submission(db_session, project.id, user_id)
    assert val2.is_valid
    assert len(val2.issues) == 0

    # Submission succeeds
    submitted = ProjectService.submit_project(db_session, project.id, user_id)
    assert submitted.status in (ProjectStatus.SUBMITTED.value, ProjectStatus.SUBMITTED)

    # Submitter can withdraw back to DRAFT
    withdrawn = ProjectService.withdraw_project(db_session, project.id, user_id)
    assert withdrawn.status == ProjectStatus.DRAFT.value
