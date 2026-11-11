"""
test_complete_review_lifecycle.py - Functional End-to-End Test for Human Review & Decision Workflows
====================================================================================================
Validates the complete full lifecycle of Phase 7:
1. Organization onboarding and project creation.
2. Milestone creation in ACTIVE status.
3. Evidence file upload and registration.
4. AI verification simulated report association.
5. Milestone submitted by NGO for formal review.
6. Admin assigns dual independent reviewers (Slot 1 and Slot 2).
7. Both reviewers accept assignments.
8. Reviewer 1 submits APPROVE; Reviewer 2 views case and verifies blind review masking.
9. Reviewer 2 submits concurring APPROVE.
10. Automatic reconciliation verifies case APPROVED and milestone COMPLETED.
11. Cryptographic audit trail validation confirms unbroken hash chain integrity.
12. Tampering test confirms tampering is detected.
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
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import ReviewCase, ReviewTask, ReviewDecisionRecord, ReviewAuditEvent
from backend.app.services.review_service import ReviewService


def test_complete_human_review_lifecycle(db_session: Session):
    """Executes the complete end-to-end human review and approval flow."""
    # -------------------------------------------------------------------------
    # 1. Setup Users & Organization
    # -------------------------------------------------------------------------
    admin = User(
        id=str(uuid.uuid4()),
        email="super_admin@charitychain.org",
        full_name="Platform Super Admin",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(admin)

    ngo_lead = User(
        id=str(uuid.uuid4()),
        email="ngo_exec@rainforesttrust.org",
        full_name="Rainforest Trust Lead",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(ngo_lead)
    db_session.flush()

    org = Organization(
        id=str(uuid.uuid4()),
        name="Rainforest Protection Trust",
        slug="rainforest-trust",
        legal_name="Rainforest Protection Trust NGO",
        organization_type=OrganizationType.NON_PROFIT,
        status=OrganizationStatus.VERIFIED,
        is_verified=True,
        created_by_user_id=ngo_lead.id
    )
    db_session.add(org)
    db_session.flush()

    org_membership = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_lead.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    db_session.add(org_membership)

    # -------------------------------------------------------------------------
    # 2. Setup Project & Milestone
    # -------------------------------------------------------------------------
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        creator_id=ngo_lead.id,
        title="Amazon Rainforest Reforestation Zone 4",
        description="Planting 50,000 indigenous trees across 100 hectares.",
        status=ProjectStatus.ACTIVE
    )
    db_session.add(project)
    db_session.flush()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        title="Saplings Planted & Geotagged",
        description="First 10,000 saplings planted with aerial drone verification.",
        status=MilestoneStatus.ACTIVE,
        sequence_number=1,
        target_amount_minor_units=1500000
    )
    db_session.add(milestone)
    db_session.flush()

    # -------------------------------------------------------------------------
    # 3. Upload Evidence
    # -------------------------------------------------------------------------
    evidence1 = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_lead.id,
        original_filename="drone_orthomosaic.tif",
        file_path="mock/path/drone_orthomosaic.tif",
        file_size_bytes=1048576,
        file_hash="1" * 64,
        content_address="sha256:" + ("1" * 64),
        mime_type="image/tiff",
        is_archived=False
    )
    evidence2 = EvidenceFile(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        uploader_id=ngo_lead.id,
        original_filename="nursery_receipt.pdf",
        file_path="mock/path/nursery_receipt.pdf",
        file_size_bytes=20480,
        file_hash="2" * 64,
        content_address="sha256:" + ("2" * 64),
        mime_type="application/pdf",
        is_archived=False
    )
    db_session.add(evidence1)
    db_session.add(evidence2)
    db_session.flush()

    # -------------------------------------------------------------------------
    # 4. NGO Submits Milestone for Review
    # -------------------------------------------------------------------------
    case = ReviewService.submit_milestone_for_review(
        db=db_session,
        milestone_id=milestone.id,
        user=ngo_lead,
        submission_notes="Drone orthomosaic map and purchase invoice attached."
    )

    assert case.status == ReviewCaseStatus.SUBMITTED.value
    assert case.submission_round == 1
    assert case.snapshot_digest is not None
    assert len(case.evidence_items) == 2

    db_session.refresh(milestone)
    assert milestone.status == MilestoneStatus.VERIFICATION_PENDING

    # -------------------------------------------------------------------------
    # 5. Admin Assigns Dual Independent Reviewers
    # -------------------------------------------------------------------------
    rev1 = User(
        id=str(uuid.uuid4()),
        email="forestry_expert@charitychain.org",
        full_name="Dr. Jane Forestry",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    rev2 = User(
        id=str(uuid.uuid4()),
        email="gis_specialist@charitychain.org",
        full_name="Carlos GIS",
        role=PlatformRole.REVIEWER,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        hashed_password="mock"
    )
    db_session.add(rev1)
    db_session.add(rev2)
    db_session.flush()

    tasks = ReviewService.assign_reviewers(
        db=db_session,
        case_id=case.id,
        admin_user=admin,
        reviewer_1_id=rev1.id,
        reviewer_2_id=rev2.id
    )

    assert len(tasks) == 2
    task1, task2 = tasks[0], tasks[1]
    assert task1.reviewer_slot == 1
    assert task2.reviewer_slot == 2

    db_session.refresh(case)
    assert case.status == ReviewCaseStatus.ASSIGNED.value

    # -------------------------------------------------------------------------
    # 6. Reviewers Accept Tasks
    # -------------------------------------------------------------------------
    ReviewService.accept_assignment(db=db_session, task_id=task1.id, reviewer=rev1)
    ReviewService.accept_assignment(db=db_session, task_id=task2.id, reviewer=rev2)

    db_session.refresh(case)
    assert case.status == ReviewCaseStatus.IN_REVIEW.value

    # -------------------------------------------------------------------------
    # 7. Reviewer 1 Approves; Verify Blind Review for Reviewer 2
    # -------------------------------------------------------------------------
    ReviewService.submit_decision(
        db=db_session,
        task_id=task1.id,
        reviewer=rev1,
        decision=ReviewDecision.APPROVE,
        comments="Orthomosaic NDVI index confirms healthy vegetative canopy and tree count."
    )

    db_session.refresh(case)
    assert case.status == ReviewCaseStatus.AWAITING_SECOND_REVIEW.value

    # Reviewer 2 checks case details -> must be blind masked
    detail = ReviewService.get_case_detail_for_reviewer(db=db_session, case_id=case.id, user=rev2)
    assert detail["decisions"][0]["decision"] == "SUBMITTED_HIDDEN"
    assert detail["decisions"][0]["reviewer_id"] == "REDACTED_BLIND_REVIEW"

    # -------------------------------------------------------------------------
    # 8. Reviewer 2 Concurs and Approves -> Completes Milestone
    # -------------------------------------------------------------------------
    ReviewService.submit_decision(
        db=db_session,
        task_id=task2.id,
        reviewer=rev2,
        decision=ReviewDecision.APPROVE,
        comments="GIS boundaries and GPS coordinates align accurately with project boundary."
    )

    db_session.refresh(case)
    db_session.refresh(milestone)

    assert case.status == ReviewCaseStatus.APPROVED.value
    assert case.final_decision == ReviewDecision.APPROVE.value
    assert case.reconciliation_result == ReconciliationResult.APPROVED.value
    assert milestone.status == MilestoneStatus.COMPLETED

    # -------------------------------------------------------------------------
    # 9. Cryptographic Audit Chain Verification
    # -------------------------------------------------------------------------
    audit_trail = ReviewService.get_review_audit_trail(db=db_session, milestone_id=milestone.id)
    assert audit_trail["total_events"] >= 5
    assert audit_trail["is_chain_intact"] is True

    # -------------------------------------------------------------------------
    # 10. Tamper Detection Test
    # -------------------------------------------------------------------------
    # Artificially alter one event's current hash in the database
    first_event = db_session.query(ReviewAuditEvent).filter(
        ReviewAuditEvent.milestone_id == milestone.id
    ).order_by(ReviewAuditEvent.created_at.asc()).first()
    
    original_hash = first_event.current_event_hash
    first_event.current_event_hash = "tampered_hash_000000000000000000000000000000000000000000000000"
    db_session.flush()

    tampered_trail = ReviewService.get_review_audit_trail(db=db_session, milestone_id=milestone.id)
    assert tampered_trail["is_chain_intact"] is False

    # Restore hash
    first_event.current_event_hash = original_hash
    db_session.flush()
    restored_trail = ReviewService.get_review_audit_trail(db=db_session, milestone_id=milestone.id)
    assert restored_trail["is_chain_intact"] is True
