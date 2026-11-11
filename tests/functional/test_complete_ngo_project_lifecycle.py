"""
test_complete_ngo_project_lifecycle.py - Functional End-to-End NGO Onboarding & Project Lifecycle
=================================================================================================
Validates the entire end-to-end journey from user registration to organization onboarding,
document upload, reviewer verification, project creation, milestone definition,
submission validation report, project approval, activation, and public discovery.
"""

import pytest
from backend.app.core.constants import (
    PlatformRole, OrganizationType, OrganizationStatus, OrganizationDocumentType,
    ProjectStatus, ProjectVisibility, OrganizationRole
)
from backend.app.api.schemas import (
    RegisterRequest, CreateOrganizationRequest, CreateProjectRequest,
    CreateMilestoneRequest, OrganizationDecisionRequest, ProjectDecisionRequest
)
from backend.app.services.auth_service import AuthService
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.integrations.storage.local import LocalStorageProvider
from backend.app.db.models.users import User, AccountStatus


def test_complete_end_to_end_ngo_project_journey(db_session, tmp_path):
    """
    Execute full Phase 3 lifecycle:
    Register User -> Create Org (DRAFT) -> Upload Doc -> Submit Org ->
    Reviewer Approves Org (VERIFIED) -> Create Project (DRAFT) ->
    Add Milestones (Budget match) -> Live Validate (13/13 pass) ->
    Submit Project (UNDER_REVIEW) -> Reviewer Approves (APPROVED) ->
    Activate Project (ACTIVE) -> Public Discovery Verification.
    """
    storage = LocalStorageProvider(str(tmp_path / "lifecycle_docs"))

    # 1. Register NGO Founder
    reg_ngo = AuthService.register_user(db_session, RegisterRequest(
        full_name="Elena Rostova",
        email="elena@arcticrescue.org",
        password="SecurePassword2026!",
        password_confirm="SecurePassword2026!"
    ))
    ngo_user = db_session.get(User, reg_ngo.user_id)
    ngo_user.is_verified = True
    ngo_user.account_status = AccountStatus.ACTIVE

    # 2. Register Independent Reviewer
    reg_rev = AuthService.register_user(db_session, RegisterRequest(
        full_name="Marcus Vance",
        email="marcus.vance@charitychain.org",
        password="SecurePassword2026!",
        password_confirm="SecurePassword2026!"
    ))
    rev_user = db_session.get(User, reg_rev.user_id)
    rev_user.is_verified = True
    rev_user.account_status = AccountStatus.ACTIVE
    rev_user.platform_role = PlatformRole.REVIEWER

    db_session.commit()

    # 3. Create Organization (Starts in DRAFT)
    org_req = CreateOrganizationRequest(
        legal_name="Arctic Wildlife Conservation Trust",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="Norwegian Directorate for Civil Protection",
        registration_number="NO-99221100",
        country_of_registration="NO",
        registered_address="Storgata 15, Tromso",
        operating_address="Storgata 15, Tromso",
        primary_operating_region="Svalbard & Arctic Sea",
        mission_statement="Protecting polar bear denning corridors from industrial disruption."
    )
    org = NGOOnboardingService.create_organization(db_session, org_req, ngo_user.id)
    assert org.status == OrganizationStatus.DRAFT.value

    # 4. Upload Verification Document (Registration Certificate)
    pdf_bytes = b"%PDF-1.4\n%Official Registration Certificate Content\n%%EOF"
    doc = NGOOnboardingService.upload_verification_document(
        db=db_session,
        organization_id=org.id,
        user_id=ngo_user.id,
        document_type=OrganizationDocumentType.REGISTRATION_CERTIFICATE.value,
        filename="arctic_trust_charity_cert.pdf",
        content=pdf_bytes,
        storage_provider=storage
    )
    assert doc.id is not None
    assert doc.file_hash_sha256 is not None

    # 5. Submit Organization for Verification
    submitted_org = NGOOnboardingService.submit_for_verification(db_session, org.id, ngo_user.id)
    assert submitted_org.status == OrganizationStatus.PENDING.value

    # 6. Reviewer Inspects Review Queue & Approves Organization
    org_queue = NGOOnboardingService.get_review_queue(db_session, reviewer_id=rev_user.id)
    assert any(item.id == org.id for item in org_queue.items)

    org_decision = OrganizationDecisionRequest(
        decision="APPROVE",
        review_notes="Verified against Norwegian entity register."
    )
    verified_org = NGOOnboardingService.review_organization(
        db=db_session,
        organization_id=org.id,
        decision=org_decision,
        reviewer_id=rev_user.id
    )
    assert verified_org.status == OrganizationStatus.VERIFIED.value

    # 7. NGO Creates Project in DRAFT
    ProjectService.seed_default_categories(db_session)
    categories = ProjectService.list_categories(db_session)
    wildlife_cat = categories[0]

    proj_req = CreateProjectRequest(
        organization_id=org.id,
        category_id=wildlife_cat.id,
        title="Polar Bear Acoustic Tracking Network",
        summary="Deploying hydrophone and satellite acoustic sensors across Svalbard sea ice corridors.",
        objectives="Map maternal polar bear movements to prevent maritime traffic collisions.",
        project_type="WILDLIFE_CONSERVATION",
        public_visibility=ProjectVisibility.PUBLIC,
        funding_target_minor=2000000,  # $20,000 USD
        currency="USD",
        country="NO",
        primary_operating_region="Svalbard Archipeligo",
        beneficiary_location="Spitsbergen Coastline",
        expected_beneficiaries=3500
    )
    project = ProjectService.create_project(db_session, proj_req, ngo_user.id)
    assert project.status == ProjectStatus.DRAFT.value

    # 8. Add Multi-Milestones (Budget match: $12k + $8k = $20k)
    m1_req = CreateMilestoneRequest(
        sequence_number=1,
        title="Deploy 12 Hydrophone Telemetry Buoys",
        objective="Deploy sensors along Hinlopen Strait",
        deliverables="Telemetry dashboard and raw acoustic data feed",
        success_criteria="Continuous audio transmission for 90 consecutive days",
        funding_target_minor=1200000,
        currency="USD",
        target_latitude=78.2232,
        target_longitude=15.6267
    )
    ProjectService.create_milestone(db_session, project.id, m1_req, ngo_user.id)

    m2_req = CreateMilestoneRequest(
        sequence_number=2,
        title="Satellite Uplink & Collision Early Warning Integration",
        objective="Integrate early warning system with regional maritime AIS receivers",
        deliverables="Automated maritime geofence alert API",
        success_criteria="Alert latency under 120 seconds for ship avoidance",
        funding_target_minor=800000,
        currency="USD",
        target_latitude=79.1000,
        target_longitude=16.0000
    )
    ProjectService.create_milestone(db_session, project.id, m2_req, ngo_user.id)

    # 9. Pre-Submission Live Validation Report
    val_report = ProjectService.validate_project_submission(db_session, project.id, ngo_user.id)
    assert val_report.is_valid is True
    assert len(val_report.issues) == 0

    # 10. NGO Submits Project for Review
    submitted_proj = ProjectService.submit_project(db_session, project.id, ngo_user.id)
    assert submitted_proj.status in (ProjectStatus.SUBMITTED.value, ProjectStatus.SUBMITTED)

    # 11. Reviewer Inspects Project Queue & Approves
    p_queue = ProjectService.get_review_queue(db_session, reviewer_id=rev_user.id)
    assert any(p.id == project.id for p in p_queue.items)

    proj_decision = ProjectDecisionRequest(
        decision="APPROVE",
        review_notes="Telemetry feasibility and marine boundary parameters confirmed."
    )
    approved_proj = ProjectService.review_project(
        db=db_session,
        project_id=project.id,
        decision=proj_decision,
        reviewer_id=rev_user.id
    )
    assert approved_proj.status == ProjectStatus.APPROVED.value

    # 12. NGO Activates Project
    active_proj = ProjectService.activate_project(db_session, project.id, ngo_user.id)
    assert active_proj.status == ProjectStatus.ACTIVE.value

    # 13. Public Donor Discovery Verification
    discovery = ProjectService.list_public_projects(
        db=db_session,
        search="Polar Bear",
        country="NO",
        page=1,
        page_size=10
    )
    assert discovery.total_count == 1
    assert discovery.items[0].id == project.id
    assert discovery.items[0].organization_name == "Arctic Wildlife Conservation Trust"
    assert discovery.items[0].funding_target_minor == 2000000
