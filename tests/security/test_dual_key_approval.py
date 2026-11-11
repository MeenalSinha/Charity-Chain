"""
test_dual_key_approval.py - Security & Concurrency Tests for Dual-Key Milestone Approval
========================================================================================
Validates that:
1. A single approval CANNOT approve a milestone or change its status to COMPLETED.
2. Dual independent approvals are strictly required for milestone completion.
3. Split decisions (disagreements) automatically escalate the case to senior adjudication.
4. The same user cannot be assigned to or decide both Slot 1 and Slot 2.
5. Blind review protections mask Reviewer 1's decision until Reviewer 2 submits theirs.
6. Decisions are completely immutable once recorded.
7. Settings validator prevents disabling dual-key approval in production mode.
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
    ReviewDecision,
    ReconciliationResult,
    MilestoneStatus,
    ProjectStatus,
    OrganizationStatus,
    OrganizationType
)
from backend.app.core.exceptions import (
    BadRequestException,
    ConflictException,
    ForbiddenException
)
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import ReviewCase, ReviewCaseEvidence, ReviewTask
from backend.app.services.review_service import ReviewService
from backend.app.core.config import Settings


@pytest.fixture
def dual_key_fixture(db_session: Session):
    """Sets up an NGO, project, milestone, review case, and two independent reviewers."""
    admin_user = User(
        id=str(uuid.uuid4()),
        email="admin_dual@charitychain.org",
        full_name="Platform Admin",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(admin_user)

    ngo_user = User(
        id=str(uuid.uuid4()),
        email="ngo_rep@charitychain.org",
        full_name="NGO Lead",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(ngo_user)
    db_session.flush()

    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Education Fund",
        slug="global-edu-fund",
        legal_name="Global Education Fund Inc",
        organization_type=OrganizationType.NON_PROFIT,
        status=OrganizationStatus.VERIFIED,
        is_verified=True,
        created_by_user_id=ngo_user.id
    )
    db_session.add(org)
    db_session.flush()

    membership = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_user.id,
        role=OrganizationRole.ADMIN,
        is_active=True
    )
    db_session.add(membership)

    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        creator_id=ngo_user.id,
        title="School Building Phase 1",
        description="Constructing primary school classrooms.",
        status=ProjectStatus.ACTIVE
    )
    db_session.add(proj)
    db_session.flush()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        title="Foundation Completed",
        description="Laying concrete foundation for classrooms.",
        status=MilestoneStatus.EVIDENCE_SUBMITTED,
        sequence_number=1,
        target_amount_minor_units=200000
    )
    db_session.add(milestone)
    db_session.flush()

    evidence = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_user.id,
        original_filename="foundation.jpg",
        file_path="mock/path/foundation.jpg",
        file_size_bytes=2048,
        file_hash="c" * 64,
        content_address="sha256:" + ("c" * 64),
        mime_type="image/jpeg",
        is_archived=False
    )
    db_session.add(evidence)
    db_session.flush()

    # Create 2 independent reviewers
    reviewer1 = User(
        id=str(uuid.uuid4()),
        email="reviewer_alpha@charitychain.org",
        full_name="Reviewer Alpha",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    reviewer2 = User(
        id=str(uuid.uuid4()),
        email="reviewer_beta@charitychain.org",
        full_name="Reviewer Beta",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer1)
    db_session.add(reviewer2)
    db_session.flush()

    # Submit review case
    case = ReviewService.submit_milestone_for_review(
        db=db_session,
        milestone_id=milestone.id,
        user=ngo_user,
        submission_notes="Foundation concrete poured and inspected."
    )

    # Assign reviewers to Slot 1 and Slot 2
    tasks = ReviewService.assign_reviewers(
        db=db_session,
        case_id=case.id,
        admin_user=admin_user,
        reviewer_1_id=reviewer1.id,
        reviewer_2_id=reviewer2.id
    )

    return {
        "admin": admin_user,
        "ngo_user": ngo_user,
        "org": org,
        "project": proj,
        "milestone": milestone,
        "case": case,
        "reviewer1": reviewer1,
        "reviewer2": reviewer2,
        "task1": tasks[0],
        "task2": tasks[1]
    }


def test_single_approval_does_not_approve_milestone(db_session: Session, dual_key_fixture):
    """Verify that when only 1 of 2 reviewers approves, milestone is NOT completed."""
    r1 = dual_key_fixture["reviewer1"]
    task1 = dual_key_fixture["task1"]
    case = dual_key_fixture["case"]
    milestone = dual_key_fixture["milestone"]

    # Reviewer 1 accepts and submits APPROVE
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.APPROVE,
        comments="Foundation meets architectural engineering specs."
    )

    # Verify case and milestone status
    db_session.refresh(case)
    db_session.refresh(milestone)

    # Case must NOT be APPROVED yet
    assert case.status == ReviewCaseStatus.AWAITING_SECOND_REVIEW.value
    assert case.reconciliation_result == ReconciliationResult.PENDING.value
    # Milestone must NOT be COMPLETED
    assert milestone.status != MilestoneStatus.COMPLETED
    assert milestone.status == MilestoneStatus.VERIFICATION_PENDING


def test_dual_approval_completes_milestone(db_session: Session, dual_key_fixture):
    """Verify that dual independent approvals approve the case and complete the milestone."""
    r1 = dual_key_fixture["reviewer1"]
    r2 = dual_key_fixture["reviewer2"]
    task1 = dual_key_fixture["task1"]
    task2 = dual_key_fixture["task2"]
    case = dual_key_fixture["case"]
    milestone = dual_key_fixture["milestone"]

    # Reviewer 1 approves
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.APPROVE,
        comments="Reviewer 1 confirms all foundation criteria are met."
    )

    # Reviewer 2 accepts and approves
    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.APPROVE,
        comments="Reviewer 2 confirms structural concrete depth verification."
    )

    db_session.refresh(case)
    db_session.refresh(milestone)

    # Dual approvals reconcile to APPROVED
    assert case.status == ReviewCaseStatus.APPROVED.value
    assert case.final_decision == ReviewDecision.APPROVE.value
    assert case.reconciliation_result == ReconciliationResult.APPROVED.value
    assert milestone.status == MilestoneStatus.COMPLETED


def test_split_decision_escalates_case(db_session: Session, dual_key_fixture):
    """Verify that split decisions (one APPROVE, one REJECT) escalate to ESCALATED status."""
    r1 = dual_key_fixture["reviewer1"]
    r2 = dual_key_fixture["reviewer2"]
    task1 = dual_key_fixture["task1"]
    task2 = dual_key_fixture["task2"]
    case = dual_key_fixture["case"]
    milestone = dual_key_fixture["milestone"]

    # Reviewer 1 approves
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.APPROVE,
        comments="Reviewer 1 approves the milestone evidence."
    )

    # Reviewer 2 rejects
    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.REJECT,
        comments="Reviewer 2 found severe discrepancy in geotagging metadata.",
        rejection_reason_category="METADATA_DISCREPANCY"
    )

    db_session.refresh(case)
    db_session.refresh(milestone)

    # Reconciles to ESCALATED
    assert case.status == ReviewCaseStatus.ESCALATED.value
    assert case.reconciliation_result == ReconciliationResult.ESCALATED.value
    assert milestone.status != MilestoneStatus.COMPLETED


def test_identical_reviewer_both_slots_rejected(db_session: Session, dual_key_fixture):
    """Verify administrator cannot assign the same user to both review slots."""
    admin = dual_key_fixture["admin"]
    r1 = dual_key_fixture["reviewer1"]
    case = dual_key_fixture["case"]

    with pytest.raises(BadRequestException) as excinfo:
        ReviewService.assign_reviewers(
            db=db_session,
            case_id=case.id,
            admin_user=admin,
            reviewer_1_id=r1.id,
            reviewer_2_id=r1.id  # Same reviewer!
        )
    assert "distinct" in str(excinfo.value).lower() or "different" in str(excinfo.value).lower()


def test_blind_review_masking(db_session: Session, dual_key_fixture):
    """Verify that Reviewer 2 cannot see Reviewer 1's decision or comments before deciding."""
    r1 = dual_key_fixture["reviewer1"]
    r2 = dual_key_fixture["reviewer2"]
    task1 = dual_key_fixture["task1"]
    task2 = dual_key_fixture["task2"]
    case = dual_key_fixture["case"]

    # Reviewer 1 submits decision
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    secret_comment = "Alpha Private Analysis: Foundation looks fully solid."
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.APPROVE,
        comments=secret_comment
    )

    # Reviewer 2 inspects case before submitting their decision
    detail = ReviewService.get_case_detail_for_reviewer(
        db=db_session,
        case_id=case.id,
        user=r2
    )

    # Must be masked for Reviewer 2
    assert len(detail["decisions"]) == 1
    masked_decision = detail["decisions"][0]
    assert masked_decision["reviewer_id"] == "REDACTED_BLIND_REVIEW"
    assert masked_decision["decision"] == "SUBMITTED_HIDDEN"
    assert secret_comment not in masked_decision["comments"]

    # Now Reviewer 2 submits their decision
    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.APPROVE,
        comments="Beta Private Analysis: Independently verified."
    )

    # After both have decided, unmasking occurs
    detail_after = ReviewService.get_case_detail_for_reviewer(
        db=db_session,
        case_id=case.id,
        user=r2
    )
    assert len(detail_after["decisions"]) == 2
    assert any(d["comments"] == secret_comment for d in detail_after["decisions"])


def test_decision_immutability(db_session: Session, dual_key_fixture):
    """Verify that once a decision is recorded, subsequent submissions fail."""
    r1 = dual_key_fixture["reviewer1"]
    task1 = dual_key_fixture["task1"]

    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.APPROVE,
        comments="First submission which is final."
    )

    # Attempt second submission on the same task
    with pytest.raises(ConflictException) as excinfo:
        ReviewService.submit_decision(
            db=db_session,
            task_id=task1.id,
            reviewer=r1,
            decision=ReviewDecision.REJECT,
            comments="Attempting to retract decision."
        )
    assert "immutable" in str(excinfo.value).lower() or "already been submitted" in str(excinfo.value).lower()


def test_production_validator_prevents_disabling_dual_key():
    """Verify Settings pydantic validator prevents disabling dual-key approval in production."""
    with pytest.raises(ValueError) as excinfo:
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="x" * 64,
            REVIEW_SECOND_REVIEW_REQUIRED=False
        )
    assert "Dual-key approval cannot be disabled in production" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo2:
        Settings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="x" * 64,
            REVIEW_REQUIRED_APPROVALS=1
        )
    assert "Dual-key approval cannot be disabled in production" in str(excinfo2.value)
