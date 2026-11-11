"""
test_reviewer_authorization.py - Security tests for Reviewer Authorization & Conflict of Interest
=================================================================================================
Verifies platform role enforcement for review queues and strict prevention
of self-review / conflict of interest by reviewers.
"""

import pytest
from backend.app.core.constants import (
    PlatformRole, OrganizationType, OrganizationStatus,
    ProjectStatus, OrganizationRole
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, CreateProjectRequest, CreateMilestoneRequest,
    OrganizationDecisionRequest, ProjectDecisionRequest, RegisterRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.auth_service import AuthService
from backend.app.core.exceptions import ForbiddenException
from backend.app.db.models.users import User, AccountStatus, Organization
from backend.app.db.models.projects import Project


@pytest.fixture
def reviewer_coi_setup(db_session, tmp_path):
    """Set up an affiliated reviewer (owns Org A) and an independent reviewer."""
    # 1. Reviewer A (also owns Org A)
    reg_rev_a = AuthService.register_user(db_session, RegisterRequest(
        full_name="Reviewer Alice",
        email="alice.coi@charitychain.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u_a = db_session.get(User, reg_rev_a.user_id)
    u_a.is_verified = True
    u_a.account_status = AccountStatus.ACTIVE
    u_a.platform_role = PlatformRole.REVIEWER

    # 2. Independent Reviewer B (has no org memberships)
    reg_rev_b = AuthService.register_user(db_session, RegisterRequest(
        full_name="Reviewer Bob",
        email="bob.indep@charitychain.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u_b = db_session.get(User, reg_rev_b.user_id)
    u_b.is_verified = True
    u_b.account_status = AccountStatus.ACTIVE
    u_b.platform_role = PlatformRole.REVIEWER

    # 3. Regular Donor
    reg_donor = AuthService.register_user(db_session, RegisterRequest(
        full_name="Dave Donor",
        email="dave.donor@example.com",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u_d = db_session.get(User, reg_donor.user_id)
    u_d.is_verified = True
    u_d.account_status = AccountStatus.ACTIVE
    u_d.platform_role = PlatformRole.DONOR

    # Reviewer A creates Org A
    org_a = NGOOnboardingService.create_organization(db_session, CreateOrganizationRequest(
        legal_name="Alice's Local Shelter",
        organization_type=OrganizationType.CHARITY,
        registration_authority="City Registry",
        registration_number="REG-ALICE",
        country_of_registration="US",
        registered_address="10 Hope Street",
        operating_address="10 Hope Street",
        primary_operating_region="Midwest",
        mission_statement="Providing emergency winter shelter."
    ), reg_rev_a.user_id)
    
    # Transition org to PENDING directly for testing review
    db_org_a = db_session.get(Organization, org_a.id)
    db_org_a.status = OrganizationStatus.PENDING

    # Setup Project under Org A
    ProjectService.seed_default_categories(db_session)
    cats = ProjectService.list_categories(db_session)
    proj_a = ProjectService.create_project(db_session, CreateProjectRequest(
        organization_id=org_a.id,
        category_id=cats[0].id,
        title="Winter Blankets & Cots",
        summary="Providing 500 insulated sleeping kits to local homeless shelters.",
        objectives="Distribute cold-weather sleeping gear before subzero temperatures.",
        funding_target_minor=500000,
        currency="USD",
        country="US",
        primary_operating_region="Midwest"
    ), reg_rev_a.user_id)
    
    # Set project to SUBMITTED
    db_p = db_session.get(Project, proj_a.id)
    db_p.status = ProjectStatus.SUBMITTED

    db_session.commit()
    return {
        "reviewer_a": reg_rev_a.user_id,
        "reviewer_b": reg_rev_b.user_id,
        "donor": reg_donor.user_id,
        "org_a": org_a.id,
        "proj_a": proj_a.id
    }


def test_regular_donor_cannot_access_review_queues(db_session, reviewer_coi_setup):
    """Verify donor cannot access reviewer queues or submit review decisions."""
    donor_id = reviewer_coi_setup["donor"]
    org_id = reviewer_coi_setup["org_a"]

    with pytest.raises(ForbiddenException):
        NGOOnboardingService.get_review_queue(db_session, page=1, page_size=10, reviewer_id=donor_id)

    with pytest.raises(ForbiddenException):
        NGOOnboardingService.review_organization(
            db=db_session,
            organization_id=org_id,
            decision=OrganizationDecisionRequest(decision="APPROVE", review_notes="Test"),
            reviewer_id=donor_id
        )


def test_reviewer_conflict_of_interest_organization_review_blocked(db_session, reviewer_coi_setup):
    """Verify Reviewer A cannot approve or reject their own organization."""
    rev_a = reviewer_coi_setup["reviewer_a"]
    org_id = reviewer_coi_setup["org_a"]

    with pytest.raises(ForbiddenException) as exc_info:
        NGOOnboardingService.review_organization(
            db=db_session,
            organization_id=org_id,
            decision=OrganizationDecisionRequest(decision="APPROVE", review_notes="Self-approval attempt"),
            reviewer_id=rev_a
        )
    assert "conflict of interest" in str(exc_info.value.message).lower()


def test_reviewer_conflict_of_interest_project_review_blocked(db_session, reviewer_coi_setup):
    """Verify Reviewer A cannot approve or reject a project belonging to their organization."""
    rev_a = reviewer_coi_setup["reviewer_a"]
    proj_id = reviewer_coi_setup["proj_a"]

    with pytest.raises(ForbiddenException) as exc_info:
        ProjectService.review_project(
            db=db_session,
            project_id=proj_id,
            decision=ProjectDecisionRequest(decision="APPROVE", review_notes="Self-approval attempt"),
            reviewer_id=rev_a
        )
    assert "conflict of interest" in str(exc_info.value.message).lower()


def test_independent_reviewer_can_approve(db_session, reviewer_coi_setup):
    """Verify independent Reviewer B (not affiliated with Org A) can review and approve."""
    rev_b = reviewer_coi_setup["reviewer_b"]
    org_id = reviewer_coi_setup["org_a"]
    proj_id = reviewer_coi_setup["proj_a"]

    # 1. Reviewer B approves Org A
    org_res = NGOOnboardingService.review_organization(
        db=db_session,
        organization_id=org_id,
        decision=OrganizationDecisionRequest(decision="APPROVE", review_notes="Legitimate documentation verified."),
        reviewer_id=rev_b
    )
    assert org_res.status == OrganizationStatus.VERIFIED.value

    # 2. Reviewer B approves Project A
    proj_res = ProjectService.review_project(
        db=db_session,
        project_id=proj_id,
        decision=ProjectDecisionRequest(decision="APPROVE", review_notes="Feasible budget and deliverables verified."),
        reviewer_id=rev_b
    )
    assert proj_res.status == ProjectStatus.APPROVED.value
