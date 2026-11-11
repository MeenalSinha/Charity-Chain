"""
test_revision_and_resubmission.py - Integration Tests for Revision & Resubmission Workflow
==========================================================================================
Validates that:
1. When reviewers request revision, structured revision requirements are validated.
2. Concordant revision requests transition the case to REVISION_REQUIRED and set a deadline.
3. NGO submits revision response with written notes and new evidence files.
4. A new ReviewCase is created for round N+1 with updated snapshot digest.
5. Previous round 1 case, tasks, and decisions remain completely intact and auditable.
6. Submissions past the revision deadline are rejected.
"""

import uuid
from datetime import datetime, timedelta
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
    ForbiddenException,
    NotFoundException
)
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import ReviewCase, ReviewCaseEvidence, ReviewTask, ReviewDecisionRecord
from backend.app.services.review_service import ReviewService


@pytest.fixture
def revision_fixture(db_session: Session):
    """Sets up an NGO, project, milestone, review case, and two reviewers."""
    admin_user = User(
        id=str(uuid.uuid4()),
        email="admin_rev@charitychain.org",
        full_name="Platform Admin",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(admin_user)

    ngo_user = User(
        id=str(uuid.uuid4()),
        email="ngo_rev@charitychain.org",
        full_name="NGO Director",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(ngo_user)
    db_session.flush()

    org = Organization(
        id=str(uuid.uuid4()),
        name="Solar for Clinics",
        slug="solar-clinics",
        legal_name="Solar for Clinics Non-Profit",
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
        role=OrganizationRole.OWNER,
        is_active=True
    )
    db_session.add(membership)

    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        creator_id=ngo_user.id,
        title="Solar Panel Installation at Rural Clinic",
        description="Installing 10kW solar system at community clinic.",
        status=ProjectStatus.ACTIVE
    )
    db_session.add(proj)
    db_session.flush()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        title="Panels Mounted on Roof",
        description="Photovoltaic panels installed and mounted.",
        status=MilestoneStatus.EVIDENCE_SUBMITTED,
        sequence_number=1,
        target_amount_minor_units=300000
    )
    db_session.add(milestone)
    db_session.flush()

    initial_evidence = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_user.id,
        original_filename="blurry_roof.jpg",
        file_path="mock/path/blurry_roof.jpg",
        file_size_bytes=1024,
        file_hash="d" * 64,
        content_address="sha256:" + ("d" * 64),
        mime_type="image/jpeg",
        is_archived=False
    )
    db_session.add(initial_evidence)
    db_session.flush()

    reviewer1 = User(
        id=str(uuid.uuid4()),
        email="rev1@charitychain.org",
        full_name="Reviewer One",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    reviewer2 = User(
        id=str(uuid.uuid4()),
        email="rev2@charitychain.org",
        full_name="Reviewer Two",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer1)
    db_session.add(reviewer2)
    db_session.flush()

    # Submit Round 1 case
    case = ReviewService.submit_milestone_for_review(
        db=db_session,
        milestone_id=milestone.id,
        user=ngo_user,
        submission_notes="Round 1: Initial panels photo attached."
    )

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
        "initial_evidence": initial_evidence,
        "case": case,
        "reviewer1": reviewer1,
        "reviewer2": reviewer2,
        "task1": tasks[0],
        "task2": tasks[1]
    }


def test_missing_structured_revision_requirements_fails(db_session: Session, revision_fixture):
    """Verify that requesting revision without structured data or sufficient rationale fails."""
    r1 = revision_fixture["reviewer1"]
    task1 = revision_fixture["task1"]

    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)

    # Submitting short comment and no structured data should fail
    with pytest.raises(BadRequestException) as excinfo:
        ReviewService.submit_decision(
            db=db_session,
            task_id=task1.id,
            reviewer=r1,
            decision=ReviewDecision.REQUEST_REVISION,
            comments="Needs a fix"  # 12 chars (>= 10 so passes min comments length, but < 15 and no structured data)
        )
    assert "revision" in str(excinfo.value).lower()


def test_dual_revision_requests_triggers_revision_required(db_session: Session, revision_fixture):
    """Verify that when both reviewers request revision, case transitions to REVISION_REQUIRED."""
    r1 = revision_fixture["reviewer1"]
    r2 = revision_fixture["reviewer2"]
    task1 = revision_fixture["task1"]
    task2 = revision_fixture["task2"]
    case = revision_fixture["case"]
    milestone = revision_fixture["milestone"]

    # Reviewer 1 requests revision
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="The roof photo is too blurry to verify panel serial numbers.",
        structured_revision_data={"required_evidence": ["High-res serial number photo", "Inverter wiring photo"]}
    )

    # Reviewer 2 requests revision
    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="Please provide a clear wide-angle photo showing all mounted panels.",
        structured_revision_data={"required_evidence": ["Wide-angle installation view"]}
    )

    db_session.refresh(case)
    db_session.refresh(milestone)

    # Case must be in REVISION_REQUIRED
    assert case.status == ReviewCaseStatus.REVISION_REQUIRED.value
    assert case.reconciliation_result == ReconciliationResult.REVISION_REQUIRED.value
    assert case.revision_deadline is not None
    assert case.revision_deadline > datetime.utcnow()
    assert milestone.status == MilestoneStatus.REVISION_REQUIRED


def test_submit_revision_response_creates_round_two(db_session: Session, revision_fixture):
    """Verify that NGO revision response creates a new submission round 2 while preserving round 1."""
    r1 = revision_fixture["reviewer1"]
    r2 = revision_fixture["reviewer2"]
    task1 = revision_fixture["task1"]
    task2 = revision_fixture["task2"]
    case = revision_fixture["case"]
    milestone = revision_fixture["milestone"]
    ngo_user = revision_fixture["ngo_user"]

    # Reviewers both request revision
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="Please upload high-resolution serial number photos.",
        structured_revision_data={"required_evidence": ["Serial numbers"]}
    )

    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="Wide angle view of complete installation is required.",
        structured_revision_data={"required_evidence": ["Wide angle view"]}
    )

    # NGO uploads new high-res photo
    new_evidence = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_user.id,
        original_filename="sharp_roof_serial.jpg",
        file_path="mock/path/sharp_roof_serial.jpg",
        file_size_bytes=4096,
        file_hash="e" * 64,
        content_address="sha256:" + ("e" * 64),
        mime_type="image/jpeg",
        is_archived=False
    )
    db_session.add(new_evidence)
    db_session.flush()

    # NGO submits revision response
    new_case = ReviewService.submit_revision_response(
        db=db_session,
        case_id=case.id,
        user=ngo_user,
        written_response="Uploaded replacement clear 4K photos showing serial numbers and wide angle.",
        new_evidence_file_ids=[new_evidence.id]
    )

    # Verify Round 2 properties
    assert new_case.id != case.id
    assert new_case.submission_round == 2
    assert new_case.status == ReviewCaseStatus.SUBMITTED.value
    assert new_case.snapshot_digest != case.snapshot_digest

    # Verify Round 1 case and decisions remain preserved (prior case marked CLOSED)
    db_session.refresh(case)
    assert case.submission_round == 1
    assert case.status == ReviewCaseStatus.CLOSED.value

    round1_decisions = db_session.query(ReviewDecisionRecord).filter(
        ReviewDecisionRecord.review_case_id == case.id
    ).all()
    assert len(round1_decisions) == 2
    assert all(d.submission_round == 1 for d in round1_decisions)


def test_expired_revision_deadline_blocks_submission(db_session: Session, revision_fixture):
    """Verify that an NGO cannot submit revision responses after the deadline has expired."""
    r1 = revision_fixture["reviewer1"]
    r2 = revision_fixture["reviewer2"]
    task1 = revision_fixture["task1"]
    task2 = revision_fixture["task2"]
    case = revision_fixture["case"]
    ngo_user = revision_fixture["ngo_user"]

    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=r1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=r1,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="Please upload clearer photos.",
        structured_revision_data={"item": 1}
    )

    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=r2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=r2,
        decision=ReviewDecision.REQUEST_REVISION,
        comments="Need better pictures please.",
        structured_revision_data={"item": 2}
    )

    # Manually expire the revision deadline
    case.revision_deadline = datetime.utcnow() - timedelta(days=2)
    db_session.flush()

    with pytest.raises(ForbiddenException) as excinfo:
        ReviewService.submit_revision_response(
            db=db_session,
            case_id=case.id,
            user=ngo_user,
            written_response="Late revision response after deadline.",
            new_evidence_file_ids=[revision_fixture["initial_evidence"].id]
        )
    assert "expired" in str(excinfo.value).lower()
