"""
reviews.py - Human Review & Decision Workflows REST API
======================================================
Endpoints for milestone review case submission, reviewer assignment,
blind independent decision submission, dual-key reconciliation,
structured revision requests, formal appeals/disputes, and cryptographic audit trails.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, Path, status, Body
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.db.session import get_db_session
from backend.app.core.permissions import (
    get_current_active_user, require_platform_role
)
from backend.app.core.constants import PlatformRole, ReviewCaseStatus
from backend.app.core.exceptions import NotFoundException, ForbiddenException
from backend.app.db.models.users import User, OrganizationMembership
from backend.app.db.models.reviews_impact import ReviewCase, ReviewDispute
from backend.app.services.review_service import ReviewService
from backend.app.api.schemas import (
    SubmitMilestoneReviewRequest, AssignReviewersRequest, TaskActionReasonRequest,
    SubmitReviewDecisionRequest, AdjudicateCaseRequest, SubmitRevisionResponseRequest,
    RaiseDisputeRequest, AddDisputeMessageRequest, ResolveDisputeRequest,
    ReviewCaseResponse, ReviewTaskResponse, ReviewDecisionResponse,
    ReviewDisputeResponse, ReviewAuditTrailResponse
)

router = APIRouter(prefix="/reviews", tags=["Human Review & Decision Governance"])


# -----------------------------------------------------------------------------
# 1. Review Case Submission & Listing
# -----------------------------------------------------------------------------

@router.post(
    "/cases/milestone/{milestone_id}/submit",
    response_model=ReviewCaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit milestone evidence package for formal human review"
)
def submit_milestone_for_review(
    milestone_id: str = Path(..., description="Milestone UUID"),
    payload: Optional[SubmitMilestoneReviewRequest] = None,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submits a milestone evidence package for human review.
    Validates submitter membership, freezes snapshot digest, and advances milestone state.
    """
    notes = payload.submission_notes if payload else None
    case = ReviewService.submit_milestone_for_review(
        db=db,
        milestone_id=milestone_id,
        user=current_user,
        submission_notes=notes
    )
    return ReviewCaseResponse(
        id=case.id,
        milestone_id=case.milestone_id,
        project_id=case.project_id,
        organization_id=case.organization_id,
        submission_round=case.submission_round,
        status=case.status,
        final_decision=case.final_decision,
        reconciliation_result=case.reconciliation_result,
        submission_notes=case.submission_notes,
        snapshot_digest=case.snapshot_digest,
        review_policy_version=case.review_policy_version,
        revision_deadline=case.revision_deadline,
        appeal_deadline=case.appeal_deadline,
        created_at=case.created_at
    )


@router.get(
    "/cases",
    response_model=List[ReviewCaseResponse],
    summary="List milestone review cases"
)
def list_review_cases(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by case status"),
    organization_id: Optional[str] = Query(None, description="Filter by organization"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Lists review cases accessible to the authenticated user.
    Admins and Reviewers see all cases; NGO members see only their organization's cases.
    """
    query = select(ReviewCase)
    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, PlatformRole.REVIEWER):
        # Scope to user's organizations
        user_orgs = db.execute(
            select(OrganizationMembership.organization_id).where(
                and_(OrganizationMembership.user_id == current_user.id, OrganizationMembership.is_active == True)
            )
        ).scalars().all()
        query = query.where(ReviewCase.organization_id.in_(user_orgs))

    if status_filter:
        query = query.where(ReviewCase.status == status_filter)
    if organization_id:
        query = query.where(ReviewCase.organization_id == organization_id)

    cases = db.execute(query.order_by(ReviewCase.created_at.desc())).scalars().all()
    return [
        ReviewCaseResponse(
            id=c.id,
            milestone_id=c.milestone_id,
            project_id=c.project_id,
            organization_id=c.organization_id,
            submission_round=c.submission_round,
            status=c.status,
            final_decision=c.final_decision,
            reconciliation_result=c.reconciliation_result,
            submission_notes=c.submission_notes,
            snapshot_digest=c.snapshot_digest,
            review_policy_version=c.review_policy_version,
            revision_deadline=c.revision_deadline,
            appeal_deadline=c.appeal_deadline,
            created_at=c.created_at
        ) for c in cases
    ]


@router.get(
    "/cases/{case_id}",
    response_model=Dict[str, Any],
    summary="Get detailed review case with blind review protection"
)
def get_review_case_detail(
    case_id: str = Path(..., description="Review case UUID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves complete case detail, evidence package, and Phase 6 AI verification reports.
    If the requesting user is Reviewer 2 and has not yet submitted, Reviewer 1's decision
    is masked to prevent anchoring bias.
    """
    return ReviewService.get_case_detail_for_reviewer(db=db, case_id=case_id, user=current_user)


# -----------------------------------------------------------------------------
# 2. Reviewer Assignment & Queue Management
# -----------------------------------------------------------------------------

@router.post(
    "/cases/{case_id}/assign",
    response_model=List[ReviewTaskResponse],
    summary="Assign dual independent reviewers to a review case"
)
def assign_reviewers(
    case_id: str = Path(..., description="Review case UUID"),
    payload: AssignReviewersRequest = ...,
    db: Session = Depends(get_db_session),
    admin_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))
):
    """
    Platform administrator assigns independent reviewers to Slot 1 and Slot 2.
    Enforces distinct identities and conflict-of-interest prohibitions.
    """
    tasks = ReviewService.assign_reviewers(
        db=db,
        case_id=case_id,
        admin_user=admin_user,
        reviewer_1_id=payload.reviewer_1_id,
        reviewer_2_id=payload.reviewer_2_id
    )
    return [
        ReviewTaskResponse(
            id=t.id,
            review_case_id=t.review_case_id,
            milestone_id=t.milestone_id,
            assigned_reviewer_id=t.assigned_reviewer_id,
            reviewer_slot=t.reviewer_slot,
            status=t.status,
            assignment_state=t.assignment_state,
            assigned_at=t.assigned_at,
            due_at=t.due_at,
            decided_at=t.decided_at
        ) for t in tasks
    ]


@router.get(
    "/my-queue",
    response_model=List[Dict[str, Any]],
    summary="Get current reviewer's assigned tasks queue"
)
def get_my_review_queue(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by assignment status"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN))
):
    """Retrieves all tasks assigned to the authenticated reviewer."""
    return ReviewService.get_reviewer_queue(db=db, reviewer=current_user, status_filter=status_filter)


@router.post(
    "/tasks/{task_id}/accept",
    response_model=ReviewTaskResponse,
    summary="Reviewer accepts assigned review task"
)
def accept_review_assignment(
    task_id: str = Path(..., description="Review task UUID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN))
):
    """Marks task accepted and transitions case status to IN_REVIEW."""
    task = ReviewService.accept_assignment(db=db, task_id=task_id, reviewer=current_user)
    return ReviewTaskResponse(
        id=task.id,
        review_case_id=task.review_case_id,
        milestone_id=task.milestone_id,
        assigned_reviewer_id=task.assigned_reviewer_id,
        reviewer_slot=task.reviewer_slot,
        status=task.status,
        assignment_state=task.assignment_state,
        assigned_at=task.assigned_at,
        due_at=task.due_at,
        decided_at=task.decided_at
    )


@router.post(
    "/tasks/{task_id}/decline",
    response_model=ReviewTaskResponse,
    summary="Reviewer declines assigned review task with reason"
)
def decline_review_assignment(
    task_id: str = Path(..., description="Review task UUID"),
    payload: TaskActionReasonRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN))
):
    """Declines task assignment with a documented reason."""
    task = ReviewService.decline_assignment(db=db, task_id=task_id, reviewer=current_user, reason=payload.reason)
    return ReviewTaskResponse(
        id=task.id,
        review_case_id=task.review_case_id,
        milestone_id=task.milestone_id,
        assigned_reviewer_id=task.assigned_reviewer_id,
        reviewer_slot=task.reviewer_slot,
        status=task.status,
        assignment_state=task.assignment_state,
        assigned_at=task.assigned_at,
        due_at=task.due_at,
        decided_at=task.decided_at
    )


@router.post(
    "/tasks/{task_id}/conflict",
    response_model=ReviewTaskResponse,
    summary="Reviewer declares conflict of interest"
)
def declare_conflict_of_interest(
    task_id: str = Path(..., description="Review task UUID"),
    payload: TaskActionReasonRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN))
):
    """Declares conflict of interest and unassigns reviewer from task."""
    task = ReviewService.declare_conflict(db=db, task_id=task_id, reviewer=current_user, reason=payload.reason)
    return ReviewTaskResponse(
        id=task.id,
        review_case_id=task.review_case_id,
        milestone_id=task.milestone_id,
        assigned_reviewer_id=task.assigned_reviewer_id,
        reviewer_slot=task.reviewer_slot,
        status=task.status,
        assignment_state=task.assignment_state,
        assigned_at=task.assigned_at,
        due_at=task.due_at,
        decided_at=task.decided_at
    )


# -----------------------------------------------------------------------------
# 3. Decision Submission & Dual-Key Sign-off
# -----------------------------------------------------------------------------

@router.post(
    "/tasks/{task_id}/decision",
    response_model=ReviewDecisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit independent review decision for an assigned slot"
)
def submit_review_decision(
    task_id: str = Path(..., description="Review task UUID"),
    payload: SubmitReviewDecisionRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN))
):
    """
    Submits an immutable decision (APPROVE, REJECT, REQUEST_REVISION).
    Automatically triggers dual-key decision reconciliation.
    """
    dec = ReviewService.submit_decision(
        db=db,
        task_id=task_id,
        reviewer=current_user,
        decision=payload.decision,
        comments=payload.comments,
        structured_revision_data=payload.structured_revision_data,
        rejection_reason_category=payload.rejection_reason_category
    )
    return ReviewDecisionResponse(
        id=dec.id,
        review_case_id=dec.review_case_id,
        task_id=dec.task_id,
        reviewer_id=dec.reviewer_id,
        decision=dec.decision.value,
        comments=dec.comments,
        submission_round=dec.submission_round,
        decided_at=dec.decided_at
    )


@router.post(
    "/cases/{case_id}/adjudicate",
    response_model=ReviewCaseResponse,
    summary="Adjudicate split or escalated review case"
)
def adjudicate_escalated_case(
    case_id: str = Path(..., description="Review case UUID"),
    payload: AdjudicateCaseRequest = Body(...),
    db: Session = Depends(get_db_session),
    admin_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))
):
    """
    Platform administrator issues final binding decision on an escalated case.
    Enforces adjudicator independence from original reviewers.
    """
    case = ReviewService.adjudicate_escalated_case(
        db=db,
        case_id=case_id,
        adjudicator=admin_user,
        decision=payload.decision,
        comments=payload.comments
    )
    return ReviewCaseResponse(
        id=case.id,
        milestone_id=case.milestone_id,
        project_id=case.project_id,
        organization_id=case.organization_id,
        submission_round=case.submission_round,
        status=case.status,
        final_decision=case.final_decision,
        reconciliation_result=case.reconciliation_result,
        submission_notes=case.submission_notes,
        snapshot_digest=case.snapshot_digest,
        review_policy_version=case.review_policy_version,
        revision_deadline=case.revision_deadline,
        appeal_deadline=case.appeal_deadline,
        created_at=case.created_at
    )


# -----------------------------------------------------------------------------
# 4. Revision Requests & NGO Resubmissions
# -----------------------------------------------------------------------------

@router.post(
    "/cases/{case_id}/revision-response",
    response_model=ReviewCaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit NGO response and replacement evidence for revision"
)
def submit_revision_response(
    case_id: str = Path(..., description="Review case UUID"),
    payload: SubmitRevisionResponseRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    NGO submits response to revision requests with replacement evidence.
    Creates new submission round N+1 for independent re-review while preserving history.
    """
    new_case = ReviewService.submit_revision_response(
        db=db,
        case_id=case_id,
        user=current_user,
        written_response=payload.written_response,
        new_evidence_file_ids=payload.new_evidence_file_ids
    )
    return ReviewCaseResponse(
        id=new_case.id,
        milestone_id=new_case.milestone_id,
        project_id=new_case.project_id,
        organization_id=new_case.organization_id,
        submission_round=new_case.submission_round,
        status=new_case.status,
        final_decision=new_case.final_decision,
        reconciliation_result=new_case.reconciliation_result,
        submission_notes=new_case.submission_notes,
        snapshot_digest=new_case.snapshot_digest,
        review_policy_version=new_case.review_policy_version,
        revision_deadline=new_case.revision_deadline,
        appeal_deadline=new_case.appeal_deadline,
        created_at=new_case.created_at
    )


# -----------------------------------------------------------------------------
# 5. Formal Disputes & Appeals
# -----------------------------------------------------------------------------

@router.post(
    "/cases/{case_id}/disputes",
    response_model=ReviewDisputeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="File formal appeal against a rejected milestone review"
)
def raise_review_dispute(
    case_id: str = Path(..., description="Review case UUID"),
    payload: RaiseDisputeRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    NGO administrator files formal appeal against rejection within configured appeal window.
    """
    dispute = ReviewService.raise_dispute(
        db=db,
        case_id=case_id,
        user=current_user,
        category=payload.dispute_category,
        reason=payload.dispute_reason,
        supporting_evidence=payload.supporting_evidence
    )
    return ReviewDisputeResponse(
        id=dispute.id,
        review_case_id=dispute.review_case_id,
        milestone_id=dispute.milestone_id,
        status=dispute.status,
        dispute_category=dispute.dispute_category,
        dispute_reason=dispute.dispute_reason,
        resolution=dispute.resolution,
        resolution_explanation=dispute.resolution_explanation,
        resolved_at=dispute.resolved_at,
        created_at=dispute.created_at
    )
    """
    NGO administrator files formal appeal against rejection within configured appeal window.
    """
    dispute = ReviewService.raise_dispute(
        db=db,
        case_id=case_id,
        user=current_user,
        category=payload.dispute_category,
        reason=payload.dispute_reason,
        supporting_evidence=payload.supporting_evidence
    )
    return ReviewDisputeResponse(
        id=dispute.id,
        review_case_id=dispute.review_case_id,
        milestone_id=dispute.milestone_id,
        status=dispute.status,
        dispute_category=dispute.dispute_category,
        dispute_reason=dispute.dispute_reason,
        resolution=dispute.resolution,
        resolution_explanation=dispute.resolution_explanation,
        resolved_at=dispute.resolved_at,
        created_at=dispute.created_at
    )


@router.get(
    "/disputes/{dispute_id}",
    response_model=Dict[str, Any],
    summary="Get dispute details and discussion messages"
)
def get_dispute_detail(
    dispute_id: str = Path(..., description="Dispute UUID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Retrieves full dispute record and message thread for authorized participants."""
    dispute = db.execute(select(ReviewDispute).where(ReviewDispute.id == dispute_id)).scalar_one_or_none()
    if not dispute:
        raise NotFoundException("Dispute not found.")

    # Authorization check
    is_admin = current_user.role == PlatformRole.PLATFORM_ADMIN
    is_adjudicator = dispute.assigned_adjudicator_id == current_user.id
    is_org_member = False
    if not is_admin and not is_adjudicator:
        mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == dispute.organization_id,
                    OrganizationMembership.user_id == current_user.id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()
        is_org_member = bool(mem)

    if not (is_admin or is_adjudicator or is_org_member):
        raise ForbiddenException("You are not authorized to view this dispute.")

    return {
        "id": dispute.id,
        "review_case_id": dispute.review_case_id,
        "milestone_id": dispute.milestone_id,
        "project_id": dispute.project_id,
        "organization_id": dispute.organization_id,
        "status": dispute.status,
        "dispute_category": dispute.dispute_category,
        "dispute_reason": dispute.dispute_reason,
        "supporting_evidence": dispute.supporting_evidence_json,
        "assigned_adjudicator_id": dispute.assigned_adjudicator_id,
        "resolution": dispute.resolution,
        "resolution_explanation": dispute.resolution_explanation,
        "resolved_at": dispute.resolved_at,
        "created_at": dispute.created_at,
        "messages": [
            {
                "id": m.id,
                "sender_id": m.sender_id,
                "sender_role": m.sender_role,
                "message": m.message,
                "created_at": m.created_at
            } for m in dispute.messages
        ]
    }


@router.post(
    "/disputes/{dispute_id}/messages",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Add message to formal dispute thread"
)
def add_dispute_message(
    dispute_id: str = Path(..., description="Dispute UUID"),
    payload: AddDisputeMessageRequest = Body(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """Appends an auditable message to the private dispute discussion thread."""
    msg = ReviewService.add_dispute_message(
        db=db,
        dispute_id=dispute_id,
        user=current_user,
        message=payload.message,
        attachments=payload.attachments
    )
    return {
        "id": msg.id,
        "dispute_id": msg.dispute_id,
        "sender_id": msg.sender_id,
        "sender_role": msg.sender_role,
        "message": msg.message,
        "created_at": msg.created_at
    }


@router.post(
    "/disputes/{dispute_id}/resolve",
    response_model=ReviewDisputeResponse,
    summary="Adjudicator issues binding dispute resolution"
)
def resolve_dispute(
    dispute_id: str = Path(..., description="Dispute UUID"),
    payload: ResolveDisputeRequest = Body(...),
    db: Session = Depends(get_db_session),
    admin_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))
):
    """Issues final binding resolution on a milestone appeal."""
    dispute = ReviewService.resolve_dispute(
        db=db,
        dispute_id=dispute_id,
        adjudicator=admin_user,
        resolution=payload.resolution,
        explanation=payload.explanation
    )
    return ReviewDisputeResponse(
        id=dispute.id,
        review_case_id=dispute.review_case_id,
        milestone_id=dispute.milestone_id,
        status=dispute.status,
        dispute_category=dispute.dispute_category,
        dispute_reason=dispute.dispute_reason,
        resolution=dispute.resolution,
        resolution_explanation=dispute.resolution_explanation,
        resolved_at=dispute.resolved_at,
        created_at=dispute.created_at
    )


# -----------------------------------------------------------------------------
# 6. Cryptographic Audit Trail
# -----------------------------------------------------------------------------

@router.get(
    "/audit/{milestone_id}",
    response_model=ReviewAuditTrailResponse,
    summary="Get cryptographic review audit trail with unbroken hash validation"
)
def get_review_audit_trail(
    milestone_id: str = Path(..., description="Milestone UUID"),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves full audit log and mathematically checks SHA-256 chain continuity.
    """
    res = ReviewService.get_review_audit_trail(db=db, milestone_id=milestone_id)
    return ReviewAuditTrailResponse(
        milestone_id=res["milestone_id"],
        total_events=res["total_events"],
        is_chain_intact=res["is_chain_intact"],
        events=res["events"]
    )
