"""
test_reviewer_assignment_and_coi.py - Unit Tests for Reviewer Assignment and Conflict-of-Interest Checks
========================================================================================================
Validates that:
1. Reviewers must be verified, active, and possess PlatformRole.REVIEWER or PLATFORM_ADMIN.
2. Reviewers who are members/owners of the organization are strictly rejected (COI Rule 1).
3. Reviewers who submitted the milestone case are strictly rejected (COI Rule 2).
4. Reviewers who uploaded any evidence file in the package are strictly rejected (COI Rule 3).
5. The same reviewer cannot occupy both Slot 1 and Slot 2 in the same case (COI Rule 4).
6. Reviewers exceeding the maximum open task workload ceiling are rejected.
7. Eligible, independent reviewers pass all checks cleanly.
"""

import uuid
from datetime import datetime
import pytest
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    PlatformRole,
    OrganizationRole,
    AccountStatus,
    ReviewCaseStatus,
    ReviewAssignmentStatus,
    MilestoneStatus,
    ProjectStatus,
    OrganizationStatus,
    OrganizationType
)
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import ReviewCase, ReviewCaseEvidence, ReviewTask
from backend.app.services.review_service import ReviewService
from backend.app.core.config import settings


@pytest.fixture
def base_coi_setup(db_session: Session):
    """Sets up an organization, project, milestone, evidence file, and review case."""
    # 1. Organization & Owner
    org_owner = User(
        id=str(uuid.uuid4()),
        email="org_owner@example.org",
        full_name="Org Owner",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock_hash"
    )
    db_session.add(org_owner)
    db_session.flush()

    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Water Global",
        slug="clean-water-global",
        legal_name="Clean Water Global NGO",
        organization_type=OrganizationType.NON_PROFIT,
        status=OrganizationStatus.VERIFIED,
        is_verified=True,
        created_by_user_id=org_owner.id
    )
    db_session.add(org)
    db_session.flush()

    membership = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=org_owner.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    db_session.add(membership)

    # 2. Project & Milestone
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        creator_id=org_owner.id,
        title="Well Construction Project",
        description="Clean water well construction project description.",
        status=ProjectStatus.ACTIVE
    )
    db_session.add(proj)
    db_session.flush()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        title="Milestone 1: Drilling Well",
        description="Drilling the water well down to 50 meters.",
        status=MilestoneStatus.EVIDENCE_SUBMITTED,
        sequence_number=1,
        target_amount_minor_units=500000
    )
    db_session.add(milestone)
    db_session.flush()

    # 3. Evidence File
    evidence = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=org_owner.id,
        original_filename="well_photo.jpg",
        file_path="mock/path/well_photo.jpg",
        file_size_bytes=1024,
        file_hash="a" * 64,
        content_address="sha256:" + ("a" * 64),
        mime_type="image/jpeg",
        is_archived=False
    )
    db_session.add(evidence)
    db_session.flush()

    # 4. Review Case
    case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=proj.id,
        organization_id=org.id,
        submitted_by_id=org_owner.id,
        submission_round=1,
        status=ReviewCaseStatus.SUBMITTED.value,
        snapshot_digest="b" * 64,
        review_policy_version="1.0"
    )
    db_session.add(case)
    db_session.flush()

    case_ev = ReviewCaseEvidence(
        id=str(uuid.uuid4()),
        review_case_id=case.id,
        evidence_file_id=evidence.id,
        evidence_version=1,
        evidence_hash=evidence.file_hash
    )
    db_session.add(case_ev)
    db_session.flush()

    return {
        "org": org,
        "org_owner": org_owner,
        "project": proj,
        "milestone": milestone,
        "evidence": evidence,
        "case": case
    }


def test_reviewer_not_found(db_session: Session, base_coi_setup):
    """Test checking eligibility of nonexistent user."""
    case = base_coi_setup["case"]
    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=str(uuid.uuid4()),
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "not found" in reason.lower()


def test_reviewer_unverified_or_inactive(db_session: Session, base_coi_setup):
    """Test reviewer with unverified email or suspended status is rejected."""
    case = base_coi_setup["case"]
    unverified_user = User(
        id=str(uuid.uuid4()),
        email="unverified@charitychain.org",
        full_name="Unverified Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=False,
        hashed_password="mock"
    )
    db_session.add(unverified_user)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=unverified_user.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "not active" in reason.lower()


def test_reviewer_invalid_role(db_session: Session, base_coi_setup):
    """Test user who is not a REVIEWER or PLATFORM_ADMIN is rejected."""
    case = base_coi_setup["case"]
    donor_user = User(
        id=str(uuid.uuid4()),
        email="donor@charitychain.org",
        full_name="Regular Donor",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(donor_user)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=donor_user.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "PlatformRole.REVIEWER" in reason


def test_coi_organization_member_rejected(db_session: Session, base_coi_setup):
    """Test reviewer who is a member of the project's organization is rejected (COI Rule 1)."""
    case = base_coi_setup["case"]
    org = base_coi_setup["org"]

    reviewer = User(
        id=str(uuid.uuid4()),
        email="coi_member@charitychain.org",
        full_name="Org Member Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer)
    db_session.flush()

    # Add active membership in the organization
    mem = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=reviewer.id,
        role=OrganizationRole.MEMBER,
        is_active=True
    )
    db_session.add(mem)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "active member/owner of the responsible organization" in reason


def test_coi_case_submitter_rejected(db_session: Session, base_coi_setup):
    """Test reviewer who submitted the case is rejected (COI Rule 2)."""
    case = base_coi_setup["case"]

    submitter_reviewer = User(
        id=str(uuid.uuid4()),
        email="submitter@charitychain.org",
        full_name="Case Submitter Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(submitter_reviewer)
    db_session.flush()

    # Update case submitter to this reviewer
    case.submitted_by_id = submitter_reviewer.id
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=submitter_reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "author of this milestone submission" in reason


def test_coi_evidence_uploader_rejected(db_session: Session, base_coi_setup):
    """Test reviewer who uploaded evidence in this case is rejected (COI Rule 3)."""
    case = base_coi_setup["case"]
    evidence = base_coi_setup["evidence"]

    uploader_reviewer = User(
        id=str(uuid.uuid4()),
        email="uploader@charitychain.org",
        full_name="Evidence Uploader Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(uploader_reviewer)
    db_session.flush()

    # Change evidence uploader to this reviewer
    evidence.uploader_id = uploader_reviewer.id
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=uploader_reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "uploaded evidence files" in reason


def test_coi_already_assigned_to_slot_rejected(db_session: Session, base_coi_setup):
    """Test reviewer already assigned to Slot 1 cannot be assigned to Slot 2 (COI Rule 4)."""
    case = base_coi_setup["case"]

    reviewer = User(
        id=str(uuid.uuid4()),
        email="dual_slot@charitychain.org",
        full_name="Dual Slot Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer)
    db_session.flush()

    # Assign reviewer to Slot 1
    task = ReviewTask(
        id=str(uuid.uuid4()),
        review_case_id=case.id,
        milestone_id=case.milestone_id,
        assigned_reviewer_id=reviewer.id,
        reviewer_slot=1,
        status="PENDING",
        assignment_state=ReviewAssignmentStatus.ASSIGNED.value
    )
    db_session.add(task)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "already assigned to another slot" in reason


def test_workload_ceiling_enforced(db_session: Session, base_coi_setup):
    """Test reviewer exceeding max open tasks workload limit is rejected."""
    case = base_coi_setup["case"]

    reviewer = User(
        id=str(uuid.uuid4()),
        email="busy_reviewer@charitychain.org",
        full_name="Busy Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer)
    db_session.flush()

    # Create max open tasks for this reviewer
    for i in range(settings.REVIEW_MAX_OPEN_TASKS_PER_REVIEWER):
        task = ReviewTask(
            id=str(uuid.uuid4()),
            milestone_id=case.milestone_id,
            assigned_reviewer_id=reviewer.id,
            reviewer_slot=1,
            status="PENDING",
            assignment_state=ReviewAssignmentStatus.ASSIGNED.value
        )
        db_session.add(task)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is False
    assert "Workload limit exceeded" in reason


def test_independent_eligible_reviewer_passes(db_session: Session, base_coi_setup):
    """Test an independent, verified reviewer with no conflicts passes all checks."""
    case = base_coi_setup["case"]

    independent_reviewer = User(
        id=str(uuid.uuid4()),
        email="independent@charitychain.org",
        full_name="Independent Reviewer",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(independent_reviewer)
    db_session.flush()

    is_eligible, reason = ReviewService.check_reviewer_eligibility_and_coi(
        db=db_session,
        reviewer_id=independent_reviewer.id,
        review_case_id=case.id
    )
    assert is_eligible is True
    assert reason == "Eligible"
