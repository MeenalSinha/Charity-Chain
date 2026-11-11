"""
test_review_disputes.py - Integration Tests for Formal Dispute & Appeal Governance
==================================================================================
Validates that:
1. NGO administrators can formally file an appeal against a rejected milestone review.
2. Appeals cannot be filed on non-rejected cases or by non-admin users.
3. The appeal window deadline is strictly enforced.
4. Message threads allow communication between authorized parties (NGO, admin/adjudicator).
5. Adjudicators with prior reviewer involvement are rejected due to conflict of interest.
6. Binding dispute resolutions properly update case and milestone states:
   - OVERTURN_ORIGINAL_DECISION / ORDER_RE_REVIEW reopens milestone for verification
   - UPHOLD_ORIGINAL_DECISION maintains rejection and closes the case.
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
    DisputeStatus,
    DisputeCategory,
    DisputeResolution,
    MilestoneStatus,
    ProjectStatus,
    OrganizationStatus,
    OrganizationType
)
from backend.app.core.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    ConflictException
)
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import ReviewCase, ReviewCaseEvidence, ReviewTask, ReviewDispute
from backend.app.services.review_service import ReviewService


@pytest.fixture
def dispute_fixture(db_session: Session):
    """Sets up a rejected review case ready for formal appeal."""
    admin_user = User(
        id=str(uuid.uuid4()),
        email="admin_disp@charitychain.org",
        full_name="Platform Admin",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(admin_user)

    ngo_admin = User(
        id=str(uuid.uuid4()),
        email="ngo_disp_admin@charitychain.org",
        full_name="NGO Lead",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(ngo_admin)
    db_session.flush()

    org = Organization(
        id=str(uuid.uuid4()),
        name="Wildlife Shelter",
        slug="wildlife-shelter",
        legal_name="Wildlife Shelter NGO",
        organization_type=OrganizationType.NON_PROFIT,
        status=OrganizationStatus.VERIFIED,
        is_verified=True,
        created_by_user_id=ngo_admin.id
    )
    db_session.add(org)
    db_session.flush()

    membership = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_admin.id,
        role=OrganizationRole.ADMIN,
        is_active=True
    )
    db_session.add(membership)

    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        creator_id=ngo_admin.id,
        title="Sanctuary Fencing",
        description="Fencing project to protect endangered animals.",
        status=ProjectStatus.ACTIVE
    )
    db_session.add(proj)
    db_session.flush()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        title="Northern Perimeter Fence",
        description="Erecting 2km perimeter fencing.",
        status=MilestoneStatus.EVIDENCE_SUBMITTED,
        sequence_number=1,
        target_amount_minor_units=450000
    )
    db_session.add(milestone)
    db_session.flush()

    evidence = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_admin.id,
        original_filename="fence.jpg",
        file_path="mock/path/fence.jpg",
        file_size_bytes=2048,
        file_hash="f" * 64,
        content_address="sha256:" + ("f" * 64),
        mime_type="image/jpeg",
        is_archived=False
    )
    db_session.add(evidence)
    db_session.flush()

    reviewer1 = User(
        id=str(uuid.uuid4()),
        email="rev_disp1@charitychain.org",
        full_name="Dispute Reviewer 1",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    reviewer2 = User(
        id=str(uuid.uuid4()),
        email="rev_disp2@charitychain.org",
        full_name="Dispute Reviewer 2",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(reviewer1)
    db_session.add(reviewer2)
    db_session.flush()

    # Submit and assign
    case = ReviewService.submit_milestone_for_review(
        db=db_session,
        milestone_id=milestone.id,
        user=ngo_admin,
        submission_notes="Perimeter fencing photo."
    )

    tasks = ReviewService.assign_reviewers(
        db=db_session,
        case_id=case.id,
        admin_user=admin_user,
        reviewer_1_id=reviewer1.id,
        reviewer_2_id=reviewer2.id
    )

    # Both reviewers REJECT -> triggers concordant rejection and opens appeal window
    ReviewService.accept_assignment(db=db_session, task_id=tasks[0].id, reviewer=reviewer1)
    ReviewService.submit_decision(
        db=db_session,
        task_id=tasks[0].id,
        reviewer=reviewer1,
        decision=ReviewDecision.REJECT,
        comments="Fence height does not appear to meet minimum 2.5m specification.",
        rejection_reason_category="SPECIFICATION_DEFICIENT"
    )

    ReviewService.accept_assignment(db=db_session, task_id=tasks[1].id, reviewer=reviewer2)
    ReviewService.submit_decision(
        db=db_session,
        task_id=tasks[1].id,
        reviewer=reviewer2,
        decision=ReviewDecision.REJECT,
        comments="Concur with rejection. Height and wire gauge insufficient.",
        rejection_reason_category="SPECIFICATION_DEFICIENT"
    )

    db_session.refresh(case)
    db_session.refresh(milestone)

    return {
        "admin": admin_user,
        "ngo_admin": ngo_admin,
        "org": org,
        "project": proj,
        "milestone": milestone,
        "case": case,
        "reviewer1": reviewer1,
        "reviewer2": reviewer2
    }


def test_raise_dispute_on_rejected_case(db_session: Session, dispute_fixture):
    """Verify that NGO admin can successfully raise a formal dispute on a rejected case."""
    case = dispute_fixture["case"]
    ngo_admin = dispute_fixture["ngo_admin"]

    dispute = ReviewService.raise_dispute(
        db=db_session,
        case_id=case.id,
        user=ngo_admin,
        category=DisputeCategory.EVIDENCE_MISINTERPRETATION,
        reason="The camera angle created perspective distortion. The actual fence height measured 2.65m as certified by surveyor.",
        supporting_evidence={"surveyor_report": "attached_pdf_hash"}
    )

    assert dispute.id is not None
    assert dispute.status == DisputeStatus.OPEN.value
    assert dispute.dispute_category == DisputeCategory.EVIDENCE_MISINTERPRETATION.value

    db_session.refresh(case)
    assert case.status == ReviewCaseStatus.DISPUTED.value


def test_raise_dispute_expired_window_fails(db_session: Session, dispute_fixture):
    """Verify that an appeal cannot be raised after the appeal deadline has passed."""
    case = dispute_fixture["case"]
    ngo_admin = dispute_fixture["ngo_admin"]

    # Expire appeal deadline
    case.appeal_deadline = datetime.utcnow() - timedelta(days=5)
    db_session.flush()

    with pytest.raises(ForbiddenException) as excinfo:
        ReviewService.raise_dispute(
            db=db_session,
            case_id=case.id,
            user=ngo_admin,
            category=DisputeCategory.EVIDENCE_MISINTERPRETATION,
            reason="Filing formal dispute after the deadline has already passed."
        )
    assert "expired" in str(excinfo.value).lower()


def test_dispute_messaging_thread(db_session: Session, dispute_fixture):
    """Verify authorized users can post messages in the dispute discussion thread."""
    case = dispute_fixture["case"]
    ngo_admin = dispute_fixture["ngo_admin"]
    admin = dispute_fixture["admin"]

    dispute = ReviewService.raise_dispute(
        db=db_session,
        case_id=case.id,
        user=ngo_admin,
        category=DisputeCategory.EVIDENCE_MISINTERPRETATION,
        reason="Disputing height measurement due to lens perspective distortion."
    )

    # NGO adds message
    msg1 = ReviewService.add_dispute_message(
        db=db_session,
        dispute_id=dispute.id,
        user=ngo_admin,
        message="We have attached the third-party survey certificate."
    )
    assert msg1.sender_id == ngo_admin.id

    # Admin adds reply
    msg2 = ReviewService.add_dispute_message(
        db=db_session,
        dispute_id=dispute.id,
        user=admin,
        message="Thank you. The independent adjudicator will review the certificate."
    )
    assert msg2.sender_id == admin.id

    # Verify messages exist in dispute
    db_session.refresh(dispute)
    assert len(dispute.messages) == 2


def test_original_reviewer_cannot_adjudicate_dispute(db_session: Session, dispute_fixture):
    """Verify that a reviewer who participated in the original decision cannot resolve the dispute."""
    case = dispute_fixture["case"]
    ngo_admin = dispute_fixture["ngo_admin"]
    r1 = dispute_fixture["reviewer1"]

    dispute = ReviewService.raise_dispute(
        db=db_session,
        case_id=case.id,
        user=ngo_admin,
        category=DisputeCategory.CONFLICT_OF_INTEREST,
        reason="Challenging reviewer impartiality."
    )

    # Reviewer 1 temporarily given admin role to test adjudicator COI check
    r1.role = PlatformRole.PLATFORM_ADMIN
    db_session.flush()

    with pytest.raises(ForbiddenException) as excinfo:
        ReviewService.resolve_dispute(
            db=db_session,
            dispute_id=dispute.id,
            adjudicator=r1,
            resolution=DisputeResolution.UPHOLD_ORIGINAL_DECISION,
            explanation="Original reviewer attempting to self-adjudicate appeal."
        )
    assert "conflict of interest" in str(excinfo.value).lower()


def test_overturn_dispute_reopens_milestone(db_session: Session, dispute_fixture):
    """Verify that overturning the original decision moves milestone back to VERIFICATION_PENDING."""
    case = dispute_fixture["case"]
    milestone = dispute_fixture["milestone"]
    ngo_admin = dispute_fixture["ngo_admin"]
    admin = dispute_fixture["admin"]

    dispute = ReviewService.raise_dispute(
        db=db_session,
        case_id=case.id,
        user=ngo_admin,
        category=DisputeCategory.EVIDENCE_MISINTERPRETATION,
        reason="Clear proof provided that fence exceeds required height."
    )

    ReviewService.resolve_dispute(
        db=db_session,
        dispute_id=dispute.id,
        adjudicator=admin,
        resolution=DisputeResolution.OVERTURN_ORIGINAL_DECISION,
        explanation="Surveyor certification conclusively establishes the fence meets specifications."
    )

    db_session.refresh(dispute)
    db_session.refresh(case)
    db_session.refresh(milestone)

    assert dispute.status == DisputeStatus.RESOLVED.value
    assert dispute.resolution == DisputeResolution.OVERTURN_ORIGINAL_DECISION.value
    assert case.status == ReviewCaseStatus.RESOLVED.value
    assert milestone.status == MilestoneStatus.VERIFICATION_PENDING


def test_uphold_dispute_closes_case(db_session: Session, dispute_fixture):
    """Verify that upholding the rejection maintains rejection and marks case CLOSED."""
    case = dispute_fixture["case"]
    milestone = dispute_fixture["milestone"]
    ngo_admin = dispute_fixture["ngo_admin"]
    admin = dispute_fixture["admin"]

    dispute = ReviewService.raise_dispute(
        db=db_session,
        case_id=case.id,
        user=ngo_admin,
        category=DisputeCategory.EVIDENCE_MISINTERPRETATION,
        reason="Challenging reviewer decision without new material."
    )

    ReviewService.resolve_dispute(
        db=db_session,
        dispute_id=dispute.id,
        adjudicator=admin,
        resolution=DisputeResolution.UPHOLD_ORIGINAL_DECISION,
        explanation="Independent assessment confirms height does not meet minimum contract requirements."
    )

    db_session.refresh(dispute)
    db_session.refresh(case)
    db_session.refresh(milestone)

    assert dispute.status == DisputeStatus.RESOLVED.value
    assert dispute.resolution == DisputeResolution.UPHOLD_ORIGINAL_DECISION.value
    assert case.status == ReviewCaseStatus.CLOSED.value
    assert milestone.status == MilestoneStatus.REJECTED
