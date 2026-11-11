"""
review_service.py - Human Review & Decision Workflows Domain Service
====================================================================
Implements review case management, independent reviewer assignment,
conflict-of-interest prevention, dual-key sign-off and reconciliation,
structured revision requests, formal disputes/appeals, and hash-chained audit trails.
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func

from backend.app.core.config import settings
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, AccountStatus, MilestoneStatus,
    ReviewDecision, ReviewCaseStatus, ReviewAssignmentStatus,
    ReconciliationResult, DisputeStatus, DisputeCategory, DisputeResolution,
    ReviewAuditEventType
)
from backend.app.core.exceptions import (
    NotFoundException, ForbiddenException, BadRequestException, ConflictException
)
from backend.app.core.state_machines import StateMachineValidator
from backend.app.core.logging import logger
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.evidence import EvidenceFile, VerificationReport
from backend.app.db.models.reviews_impact import (
    ReviewCase, ReviewCaseEvidence, ReviewTask, ReviewDecisionRecord,
    ReviewDispute, ReviewDisputeMessage, ReviewAuditEvent,
    compute_review_audit_hash
)
from backend.app.integrations.storage.factory import get_storage_provider


class ReviewService:
    """Core domain service for human review, dual-key sign-off, and dispute governance."""

    # -------------------------------------------------------------------------
    # Audit Trail Helper
    # -------------------------------------------------------------------------

    @classmethod
    def _append_audit_event(
        cls,
        db: Session,
        milestone_id: str,
        actor_id: Optional[str],
        event_type: str,
        details: Dict[str, Any],
        review_case_id: Optional[str] = None
    ) -> ReviewAuditEvent:
        """Appends an event to the tamper-evident SHA-256 audit hash chain."""
        last_event = db.execute(
            select(ReviewAuditEvent)
            .where(ReviewAuditEvent.milestone_id == milestone_id)
            .order_by(ReviewAuditEvent.created_at.desc())
        ).scalars().first()

        prev_hash = last_event.current_event_hash if last_event else ("0" * 64)
        curr_hash = compute_review_audit_hash(
            previous_hash=prev_hash,
            milestone_id=milestone_id,
            actor_id=actor_id,
            event_type=event_type,
            details=details
        )

        audit = ReviewAuditEvent(
            id=str(uuid.uuid4()),
            review_case_id=review_case_id,
            milestone_id=milestone_id,
            actor_id=actor_id,
            event_type=event_type,
            details_json=details,
            previous_event_hash=prev_hash,
            current_event_hash=curr_hash,
            created_at=datetime.utcnow()
        )
        db.add(audit)
        db.flush()
        return audit

    # -------------------------------------------------------------------------
    # 1. Review Case Creation & Submission
    # -------------------------------------------------------------------------

    @classmethod
    def submit_milestone_for_review(
        cls,
        db: Session,
        milestone_id: str,
        user: User,
        submission_notes: Optional[str] = None
    ) -> ReviewCase:
        """
        Submits an NGO's milestone evidence package for formal human review.
        Validates RBAC, freezes a cryptographic snapshot digest, binds exact evidence
        hashes and AI reports, and initializes submission round 1.
        """
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException(f"Milestone '{milestone_id}' not found.")

        project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none()
        if not project:
            raise NotFoundException("Associated project not found.")

        # Validate submitter authorization: must belong to the project organization
        if user.role != PlatformRole.PLATFORM_ADMIN:
            membership = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == project.organization_id,
                        OrganizationMembership.user_id == user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            if not membership or membership.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER):
                raise ForbiddenException("You do not have permission to submit milestones for this organization.")

        # Validate milestone status
        valid_statuses = (MilestoneStatus.ACTIVE, MilestoneStatus.EVIDENCE_SUBMITTED, MilestoneStatus.REVISION_REQUIRED)
        if milestone.status not in valid_statuses:
            raise BadRequestException(
                f"Milestone cannot be submitted for review from status '{milestone.status.value}'. Must be ACTIVE, EVIDENCE_SUBMITTED, or REVISION_REQUIRED."
            )

        # Collect active evidence files
        evidence_files = db.execute(
            select(EvidenceFile).where(
                and_(EvidenceFile.milestone_id == milestone.id, EvidenceFile.is_archived == False)
            )
        ).scalars().all()

        if not evidence_files:
            raise BadRequestException("Cannot submit milestone for review without uploaded evidence files.")

        # Verify evidence integrity & resolve latest Phase 6 AI reports
        evidence_bindings = []
        digest_components = []

        storage = get_storage_provider()
        for ev in evidence_files:
            # Check physical file integrity if required
            if settings.REVIEW_REQUIRE_EVIDENCE_INTEGRITY:
                try:
                    file_bytes = storage.get_file(ev.storage_key)
                    computed_hash = hashlib.sha256(file_bytes).hexdigest()
                    if computed_hash != ev.file_hash:
                        raise ConflictException(f"Evidence file '{ev.original_filename}' failed cryptographic integrity check.")
                except Exception as e:
                    logger.warning(f"Storage integrity check error for '{ev.id}': {e}")

            # Retrieve latest AI verification report
            ai_report = db.execute(
                select(VerificationReport)
                .where(VerificationReport.evidence_file_id == ev.id)
                .order_by(VerificationReport.processing_completed_at.desc())
            ).scalars().first()

            if settings.REVIEW_REQUIRE_COMPLETED_AI_REPORT and not ai_report:
                logger.info(f"Submitting milestone {milestone_id} without completed AI report for file {ev.id}")

            report_id = ai_report.id if ai_report else None
            pipeline_ver = ai_report.pipeline_version if ai_report else "NONE"

            evidence_bindings.append({
                "evidence_file_id": ev.id,
                "evidence_version": ev.version_number,
                "evidence_hash": ev.file_hash,
                "verification_report_id": report_id,
                "ai_pipeline_version": pipeline_ver,
                "evidence_role": "PRIMARY"
            })
            digest_components.append(f"{ev.id}:{ev.version_number}:{ev.file_hash}:{report_id or 'none'}")

        # Compute frozen snapshot digest
        snapshot_digest = hashlib.sha256("|".join(sorted(digest_components)).encode("utf-8")).hexdigest()

        # Compute submission round
        last_case = db.execute(
            select(ReviewCase)
            .where(ReviewCase.milestone_id == milestone.id)
            .order_by(ReviewCase.submission_round.desc())
        ).scalars().first()
        submission_round = (last_case.submission_round + 1) if last_case else 1

        # Create ReviewCase
        review_case = ReviewCase(
            id=str(uuid.uuid4()),
            milestone_id=milestone.id,
            project_id=project.id,
            organization_id=project.organization_id,
            submission_round=submission_round,
            submitted_by_id=user.id,
            status=ReviewCaseStatus.SUBMITTED.value,
            submission_notes=submission_notes,
            snapshot_digest=snapshot_digest,
            snapshot_data_json={
                "milestone_title": milestone.title,
                "project_title": project.title,
                "evidence_count": len(evidence_files),
                "submission_round": submission_round,
                "digest": snapshot_digest
            },
            review_policy_version=settings.REVIEW_POLICY_VERSION,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(review_case)
        db.flush()

        # Create ReviewCaseEvidence items
        for binding in evidence_bindings:
            rce = ReviewCaseEvidence(
                id=str(uuid.uuid4()),
                review_case_id=review_case.id,
                evidence_file_id=binding["evidence_file_id"],
                evidence_version=binding["evidence_version"],
                evidence_hash=binding["evidence_hash"],
                verification_report_id=binding["verification_report_id"],
                ai_pipeline_version=binding["ai_pipeline_version"],
                evidence_role=binding["evidence_role"],
                snapshot_timestamp=datetime.utcnow()
            )
            db.add(rce)

        # Update milestone status to VERIFICATION_PENDING
        milestone.status = MilestoneStatus.VERIFICATION_PENDING

        # Append audit log
        cls._append_audit_event(
            db=db,
            milestone_id=milestone.id,
            actor_id=user.id,
            event_type=ReviewAuditEventType.CASE_CREATED.value,
            details={
                "action": "SUBMIT_MILESTONE_FOR_REVIEW",
                "review_case_id": review_case.id,
                "submission_round": submission_round,
                "snapshot_digest": snapshot_digest,
                "evidence_count": len(evidence_files)
            },
            review_case_id=review_case.id
        )

        db.commit()
        return review_case

    # -------------------------------------------------------------------------
    # 2. Reviewer Eligibility & Conflict-of-Interest (COI) Control
    # -------------------------------------------------------------------------

    @classmethod
    def check_reviewer_eligibility_and_coi(
        cls,
        db: Session,
        reviewer_id: str,
        review_case_id: str
    ) -> Tuple[bool, str]:
        """
        Validates reviewer active status, role, workload ceiling, and enforces
        strict conflict-of-interest prohibitions.
        """
        reviewer = db.execute(select(User).where(User.id == reviewer_id)).scalar_one_or_none()
        if not reviewer:
            return False, "Reviewer account not found."

        if not reviewer.is_verified or reviewer.account_status != AccountStatus.ACTIVE:
            return False, f"Reviewer account is not active (status: {reviewer.account_status.value})."

        if reviewer.role not in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
            return False, f"User does not possess PlatformRole.REVIEWER (role: {reviewer.role.value})."

        case = db.execute(select(ReviewCase).where(ReviewCase.id == review_case_id)).scalar_one_or_none()
        if not case:
            return False, "Review case not found."

        # COI Rule 1: Reviewer cannot belong to the responsible organization
        membership = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == case.organization_id,
                    OrganizationMembership.user_id == reviewer.id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()
        if membership:
            return False, "Conflict of interest: Reviewer is an active member/owner of the responsible organization."

        # COI Rule 2: Reviewer cannot be the submitter of the review case
        if case.submitted_by_id == reviewer.id:
            return False, "Conflict of interest: Reviewer is the author of this milestone submission."

        # COI Rule 3: Reviewer cannot have uploaded any of the submitted evidence files
        case_ev_ids = [e.evidence_file_id for e in case.evidence_items]
        uploader_match = db.execute(
            select(EvidenceFile).where(
                and_(EvidenceFile.id.in_(case_ev_ids), EvidenceFile.uploader_id == reviewer.id)
            )
        ).scalars().first()
        if uploader_match:
            return False, "Conflict of interest: Reviewer uploaded evidence files included in this case."

        # COI Rule 4: Reviewer cannot already have decided or been assigned to the other slot in this case
        existing_task = db.execute(
            select(ReviewTask).where(
                and_(
                    ReviewTask.review_case_id == case.id,
                    ReviewTask.assigned_reviewer_id == reviewer.id,
                    ReviewTask.assignment_state != ReviewAssignmentStatus.DECLINED.value
                )
            )
        ).scalar_one_or_none()
        if existing_task:
            return False, "Conflict of interest: Reviewer is already assigned to another slot in this case."

        # Workload check
        active_tasks_count = db.execute(
            select(func.count(ReviewTask.id)).where(
                and_(
                    ReviewTask.assigned_reviewer_id == reviewer.id,
                    ReviewTask.assignment_state.in_([
                        ReviewAssignmentStatus.ASSIGNED.value,
                        ReviewAssignmentStatus.ACCEPTED.value,
                        ReviewAssignmentStatus.IN_PROGRESS.value
                    ])
                )
            )
        ).scalar() or 0

        if active_tasks_count >= settings.REVIEW_MAX_OPEN_TASKS_PER_REVIEWER:
            return False, f"Workload limit exceeded ({active_tasks_count}/{settings.REVIEW_MAX_OPEN_TASKS_PER_REVIEWER} active tasks)."

        return True, "Eligible"

    # -------------------------------------------------------------------------
    # 3. Reviewer Assignment & Task Lifecycle
    # -------------------------------------------------------------------------

    @classmethod
    def assign_reviewers(
        cls,
        db: Session,
        case_id: str,
        admin_user: User,
        reviewer_1_id: str,
        reviewer_2_id: Optional[str] = None
    ) -> List[ReviewTask]:
        """
        Assigns independent reviewers to Slot 1 (Reviewer A) and Slot 2 (Reviewer B).
        Enforces distinct identities, conflict-of-interest checks, and due dates.
        """
        if admin_user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only platform administrators can assign reviewers.")

        case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException(f"Review case '{case_id}' not found.")

        if case.status in (ReviewCaseStatus.APPROVED.value, ReviewCaseStatus.CLOSED.value, ReviewCaseStatus.CANCELLED.value):
            raise BadRequestException(f"Cannot assign reviewers to a finalized case (status: {case.status}).")

        # Dual-key policy requires two distinct reviewers
        if settings.REVIEW_SECOND_REVIEW_REQUIRED and not reviewer_2_id:
            raise BadRequestException("Dual-key policy requires two independent reviewers to be assigned.")

        if reviewer_2_id and reviewer_1_id == reviewer_2_id:
            raise BadRequestException("Dual-key violation: Reviewer 1 and Reviewer 2 must be distinct users.")

        # Check COI for Reviewer 1
        ok1, reason1 = cls.check_reviewer_eligibility_and_coi(db, reviewer_1_id, case.id)
        if not ok1:
            raise BadRequestException(f"Reviewer 1 ineligible: {reason1}")

        # Check COI for Reviewer 2
        if reviewer_2_id:
            ok2, reason2 = cls.check_reviewer_eligibility_and_coi(db, reviewer_2_id, case.id)
            if not ok2:
                raise BadRequestException(f"Reviewer 2 ineligible: {reason2}")

        due_date = datetime.utcnow() + timedelta(hours=settings.REVIEW_TASK_DUE_HOURS)
        tasks = []

        # Cancel any previous unfulfilled assignments for this case
        prev_tasks = db.execute(
            select(ReviewTask).where(
                and_(
                    ReviewTask.review_case_id == case.id,
                    ReviewTask.assignment_state.in_([
                        ReviewAssignmentStatus.ASSIGNED.value,
                        ReviewAssignmentStatus.ACCEPTED.value,
                        ReviewAssignmentStatus.IN_PROGRESS.value
                    ])
                )
            )
        ).scalars().all()
        for pt in prev_tasks:
            pt.assignment_state = ReviewAssignmentStatus.CANCELLED.value

        # Slot 1 Task
        t1 = ReviewTask(
            id=str(uuid.uuid4()),
            review_case_id=case.id,
            milestone_id=case.milestone_id,
            assigned_reviewer_id=reviewer_1_id,
            reviewer_slot=1,
            status=ReviewAssignmentStatus.ASSIGNED.value,
            assignment_state=ReviewAssignmentStatus.ASSIGNED.value,
            assigned_at=datetime.utcnow(),
            due_at=due_date
        )
        db.add(t1)
        tasks.append(t1)

        # Slot 2 Task
        if reviewer_2_id:
            t2 = ReviewTask(
                id=str(uuid.uuid4()),
                review_case_id=case.id,
                milestone_id=case.milestone_id,
                assigned_reviewer_id=reviewer_2_id,
                reviewer_slot=2,
                status=ReviewAssignmentStatus.ASSIGNED.value,
                assignment_state=ReviewAssignmentStatus.ASSIGNED.value,
                assigned_at=datetime.utcnow(),
                due_at=due_date
            )
            db.add(t2)
            tasks.append(t2)

        case.status = ReviewCaseStatus.ASSIGNED.value
        case.updated_at = datetime.utcnow()

        cls._append_audit_event(
            db=db,
            milestone_id=case.milestone_id,
            actor_id=admin_user.id,
            event_type=ReviewAuditEventType.REVIEWER_ASSIGNED.value,
            details={
                "action": "ASSIGN_REVIEWERS",
                "case_id": case.id,
                "reviewer_1_id": reviewer_1_id,
                "reviewer_2_id": reviewer_2_id,
                "due_at": due_date.isoformat()
            },
            review_case_id=case.id
        )

        db.commit()
        return tasks

    @classmethod
    def accept_assignment(cls, db: Session, task_id: str, reviewer: User) -> ReviewTask:
        """Reviewer acknowledges and accepts an assigned review task."""
        task = db.execute(select(ReviewTask).where(ReviewTask.id == task_id)).scalar_one_or_none()
        if not task:
            raise NotFoundException("Review task not found.")
        if task.assigned_reviewer_id != reviewer.id and reviewer.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("You are not assigned to this review task.")

        StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus(task.assignment_state),
            ReviewAssignmentStatus.ACCEPTED
        )
        task.assignment_state = ReviewAssignmentStatus.ACCEPTED.value
        task.accepted_at = datetime.utcnow()

        case = db.execute(select(ReviewCase).where(ReviewCase.id == task.review_case_id)).scalar_one_or_none()
        if case and case.status == ReviewCaseStatus.ASSIGNED.value:
            case.status = ReviewCaseStatus.IN_REVIEW.value

        cls._append_audit_event(
            db=db,
            milestone_id=task.milestone_id,
            actor_id=reviewer.id,
            event_type=ReviewAuditEventType.ASSIGNMENT_ACCEPTED.value,
            details={"task_id": task.id, "slot": task.reviewer_slot},
            review_case_id=task.review_case_id
        )
        db.commit()
        return task

    @classmethod
    def decline_assignment(cls, db: Session, task_id: str, reviewer: User, reason: str) -> ReviewTask:
        """Reviewer declines an assigned task with a required explanation."""
        if not reason or len(reason.strip()) < 5:
            raise BadRequestException("A detailed reason is required to decline a review assignment.")

        task = db.execute(select(ReviewTask).where(ReviewTask.id == task_id)).scalar_one_or_none()
        if not task:
            raise NotFoundException("Review task not found.")
        if task.assigned_reviewer_id != reviewer.id and reviewer.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("You are not assigned to this review task.")

        StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus(task.assignment_state),
            ReviewAssignmentStatus.DECLINED
        )
        task.assignment_state = ReviewAssignmentStatus.DECLINED.value
        task.decline_reason = reason.strip()

        cls._append_audit_event(
            db=db,
            milestone_id=task.milestone_id,
            actor_id=reviewer.id,
            event_type=ReviewAuditEventType.ASSIGNMENT_DECLINED.value,
            details={"task_id": task.id, "slot": task.reviewer_slot, "reason": reason.strip()},
            review_case_id=task.review_case_id
        )
        db.commit()
        return task

    @classmethod
    def declare_conflict(cls, db: Session, task_id: str, reviewer: User, reason: str) -> ReviewTask:
        """Reviewer declares a conflict of interest, removing themselves from the task."""
        if not reason or len(reason.strip()) < 5:
            raise BadRequestException("A specific conflict explanation is required.")

        task = db.execute(select(ReviewTask).where(ReviewTask.id == task_id)).scalar_one_or_none()
        if not task:
            raise NotFoundException("Review task not found.")
        if task.assigned_reviewer_id != reviewer.id and reviewer.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("You are not assigned to this review task.")

        task.assignment_state = ReviewAssignmentStatus.DECLINED.value
        task.conflict_declared = True
        task.conflict_reason = reason.strip()

        cls._append_audit_event(
            db=db,
            milestone_id=task.milestone_id,
            actor_id=reviewer.id,
            event_type=ReviewAuditEventType.CONFLICT_DECLARED.value,
            details={"task_id": task.id, "slot": task.reviewer_slot, "reason": reason.strip()},
            review_case_id=task.review_case_id
        )
        db.commit()
        return task

    # -------------------------------------------------------------------------
    # 4. Independent Decision Submission & Dual-Key Reconciliation
    # -------------------------------------------------------------------------

    @classmethod
    def submit_decision(
        cls,
        db: Session,
        task_id: str,
        reviewer: User,
        decision: ReviewDecision,
        comments: str,
        structured_revision_data: Optional[Dict[str, Any]] = None,
        rejection_reason_category: Optional[str] = None
    ) -> ReviewDecisionRecord:
        """
        Records an immutable review decision for a task slot.
        Enforces dual-key independence, rechecks COI, binds exact snapshot digest,
        and triggers automatic decision reconciliation.
        """
        task = db.execute(select(ReviewTask).where(ReviewTask.id == task_id)).scalar_one_or_none()
        if not task:
            raise NotFoundException("Review task not found.")

        if task.assigned_reviewer_id != reviewer.id and reviewer.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("You are not assigned to this review task.")

        if task.assignment_state == ReviewAssignmentStatus.DECIDED.value:
            raise ConflictException("Decision has already been submitted for this task slot. Decisions are immutable.")

        case = db.execute(select(ReviewCase).where(ReviewCase.id == task.review_case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException("Associated review case not found.")

        # Re-check COI at decision submission time
        ok, coi_msg = cls.check_reviewer_eligibility_and_coi(db, reviewer.id, case.id)
        # Note: If reviewer is already assigned to THIS task, eligibility check allows it
        # but prevents cross-slot dual occupancy
        other_slot_decided = db.execute(
            select(ReviewDecisionRecord).where(
                and_(
                    ReviewDecisionRecord.review_case_id == case.id,
                    ReviewDecisionRecord.reviewer_id == reviewer.id,
                    ReviewDecisionRecord.submission_round == case.submission_round
                )
            )
        ).scalar_one_or_none()
        if other_slot_decided:
            raise ForbiddenException("Dual-key violation: You have already submitted a decision for the other review slot in this case.")

        # Validate comments and structured requirements
        if not comments or len(comments.strip()) < 10:
            raise BadRequestException("Detailed review comments (minimum 10 characters) are required.")

        if decision == ReviewDecision.REQUEST_REVISION and not structured_revision_data and len(comments.strip()) < 15:
            raise BadRequestException("Actionable revision requirements must be documented when requesting revisions.")

        if decision == ReviewDecision.REJECT and not rejection_reason_category:
            rejection_reason_category = "EVIDENCE_INSUFFICIENT"

        # Create immutable decision record
        decision_rec = ReviewDecisionRecord(
            id=str(uuid.uuid4()),
            review_case_id=case.id,
            task_id=task.id,
            reviewer_id=reviewer.id,
            decision=decision,
            comments=comments.strip(),
            structured_revision_data_json=structured_revision_data,
            rejection_reason_category=rejection_reason_category,
            evidence_snapshot_digest=case.snapshot_digest,
            submission_round=case.submission_round,
            decided_at=datetime.utcnow()
        )
        db.add(decision_rec)

        # Mark task decided
        task.assignment_state = ReviewAssignmentStatus.DECIDED.value
        task.status = "COMPLETED"
        task.decided_at = datetime.utcnow()

        cls._append_audit_event(
            db=db,
            milestone_id=case.milestone_id,
            actor_id=reviewer.id,
            event_type=ReviewAuditEventType.DECISION_SUBMITTED.value,
            details={
                "task_id": task.id,
                "slot": task.reviewer_slot,
                "decision": decision.value,
                "comments": comments.strip()[:200]
            },
            review_case_id=case.id
        )

        db.flush()

        # Reconcile dual-key decisions
        cls.reconcile_case_decisions(db=db, case_id=case.id)
        db.commit()
        return decision_rec

    @classmethod
    def reconcile_case_decisions(cls, db: Session, case_id: str) -> ReconciliationResult:
        """
        Reconciles decisions across independent review slots for the active submission round:
        - 1 of 2 decided: Case moves to AWAITING_SECOND_REVIEW (blind review preserved).
        - Both APPROVE: Milestone approved & completed.
        - Both REQUEST_REVISION: Milestone requires revision; NGO given deadline.
        - Both REJECT: Milestone rejected; NGO given appeal deadline.
        - Disagree (split decision): ESCALATED to independent senior adjudicator.
        """
        case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException("Review case not found.")

        milestone = db.execute(select(Milestone).where(Milestone.id == case.milestone_id)).scalar_one_or_none()

        # Fetch active decisions for this submission round
        decisions = db.execute(
            select(ReviewDecisionRecord).where(
                and_(
                    ReviewDecisionRecord.review_case_id == case.id,
                    ReviewDecisionRecord.submission_round == case.submission_round,
                    ReviewDecisionRecord.is_superseded == False
                )
            )
        ).scalars().all()

        if len(decisions) == 0:
            return ReconciliationResult.PENDING

        if len(decisions) == 1 and settings.REVIEW_SECOND_REVIEW_REQUIRED:
            case.status = ReviewCaseStatus.AWAITING_SECOND_REVIEW.value
            case.reconciliation_result = ReconciliationResult.PENDING.value
            case.updated_at = datetime.utcnow()
            return ReconciliationResult.PENDING

        dec_values = [d.decision for d in decisions]

        # Case 1: Both APPROVE
        if all(d == ReviewDecision.APPROVE for d in dec_values) and len(dec_values) >= settings.REVIEW_REQUIRED_APPROVALS:
            case.status = ReviewCaseStatus.APPROVED.value
            case.final_decision = ReviewDecision.APPROVE.value
            case.reconciliation_result = ReconciliationResult.APPROVED.value
            case.finalized_at = datetime.utcnow()

            # State machine update: Complete milestone
            if milestone:
                milestone.status = MilestoneStatus.COMPLETED

            cls._append_audit_event(
                db=db,
                milestone_id=case.milestone_id,
                actor_id=None,
                event_type=ReviewAuditEventType.DECISION_RECONCILED.value,
                details={"result": "APPROVED", "decisions_count": len(dec_values)},
                review_case_id=case.id
            )
            return ReconciliationResult.APPROVED

        # Case 2: Both REQUEST_REVISION
        elif all(d == ReviewDecision.REQUEST_REVISION for d in dec_values):
            case.status = ReviewCaseStatus.REVISION_REQUIRED.value
            case.final_decision = ReviewDecision.REQUEST_REVISION.value
            case.reconciliation_result = ReconciliationResult.REVISION_REQUIRED.value
            case.revision_deadline = datetime.utcnow() + timedelta(hours=settings.REVIEW_REVISION_DUE_HOURS)

            if milestone:
                milestone.status = MilestoneStatus.REVISION_REQUIRED

            cls._append_audit_event(
                db=db,
                milestone_id=case.milestone_id,
                actor_id=None,
                event_type=ReviewAuditEventType.REVISION_REQUESTED.value,
                details={"result": "REVISION_REQUIRED", "deadline": case.revision_deadline.isoformat()},
                review_case_id=case.id
            )
            return ReconciliationResult.REVISION_REQUIRED

        # Case 3: Both REJECT
        elif all(d == ReviewDecision.REJECT for d in dec_values):
            case.status = ReviewCaseStatus.REJECTED.value
            case.final_decision = ReviewDecision.REJECT.value
            case.reconciliation_result = ReconciliationResult.REJECTED.value
            case.appeal_deadline = datetime.utcnow() + timedelta(hours=settings.REVIEW_APPEAL_WINDOW_HOURS)
            case.finalized_at = datetime.utcnow()

            if milestone:
                milestone.status = MilestoneStatus.REJECTED

            cls._append_audit_event(
                db=db,
                milestone_id=case.milestone_id,
                actor_id=None,
                event_type=ReviewAuditEventType.DECISION_RECONCILED.value,
                details={"result": "REJECTED", "appeal_deadline": case.appeal_deadline.isoformat()},
                review_case_id=case.id
            )
            return ReconciliationResult.REJECTED

        # Case 4: Reviewers Disagree -> ESCALATED
        else:
            case.status = ReviewCaseStatus.ESCALATED.value
            case.reconciliation_result = ReconciliationResult.ESCALATED.value
            cls._append_audit_event(
                db=db,
                milestone_id=case.milestone_id,
                actor_id=None,
                event_type=ReviewAuditEventType.ESCALATED.value,
                details={
                    "result": "ESCALATED",
                    "decisions": [d.value for d in dec_values],
                    "message": "Independent reviewer decisions split. Escalated for administrative adjudication."
                },
                review_case_id=case.id
            )
            return ReconciliationResult.ESCALATED

    # -------------------------------------------------------------------------
    # 5. Adjudication of Escalated Cases
    # -------------------------------------------------------------------------

    @classmethod
    def adjudicate_escalated_case(
        cls,
        db: Session,
        case_id: str,
        adjudicator: User,
        decision: ReviewDecision,
        comments: str
    ) -> ReviewCase:
        """
        Senior reviewer or platform admin breaks deadlock on escalated split reviews.
        Enforces adjudicator independence from original reviewers and responsible NGO.
        """
        if adjudicator.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only platform administrators may adjudicate escalated review cases.")

        case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException("Review case not found.")

        if case.status != ReviewCaseStatus.ESCALATED.value:
            raise BadRequestException(f"Case cannot be adjudicated from status '{case.status}'. Must be ESCALATED.")

        # Adjudicator COI: Cannot be one of the original reviewers in this case
        prior_reviewers = db.execute(
            select(ReviewDecisionRecord.reviewer_id).where(ReviewDecisionRecord.review_case_id == case.id)
        ).scalars().all()
        if adjudicator.id in prior_reviewers:
            raise ForbiddenException("Adjudicator conflict of interest: Cannot adjudicate a case you personally reviewed.")

        if not comments or len(comments.strip()) < 15:
            raise BadRequestException("Detailed adjudication rationale (minimum 15 characters) is required.")

        milestone = db.execute(select(Milestone).where(Milestone.id == case.milestone_id)).scalar_one_or_none()

        case.adjudicator_id = adjudicator.id
        case.adjudication_comments = comments.strip()
        case.adjudicated_at = datetime.utcnow()
        case.final_decision = decision.value

        if decision == ReviewDecision.APPROVE:
            case.status = ReviewCaseStatus.APPROVED.value
            case.reconciliation_result = ReconciliationResult.APPROVED.value
            case.finalized_at = datetime.utcnow()
            if milestone:
                milestone.status = MilestoneStatus.COMPLETED

        elif decision == ReviewDecision.REQUEST_REVISION:
            case.status = ReviewCaseStatus.REVISION_REQUIRED.value
            case.reconciliation_result = ReconciliationResult.REVISION_REQUIRED.value
            case.revision_deadline = datetime.utcnow() + timedelta(hours=settings.REVIEW_REVISION_DUE_HOURS)
            if milestone:
                milestone.status = MilestoneStatus.REVISION_REQUIRED

        elif decision == ReviewDecision.REJECT:
            case.status = ReviewCaseStatus.REJECTED.value
            case.reconciliation_result = ReconciliationResult.REJECTED.value
            case.appeal_deadline = datetime.utcnow() + timedelta(hours=settings.REVIEW_APPEAL_WINDOW_HOURS)
            case.finalized_at = datetime.utcnow()
            if milestone:
                milestone.status = MilestoneStatus.REJECTED

        cls._append_audit_event(
            db=db,
            milestone_id=case.milestone_id,
            actor_id=adjudicator.id,
            event_type=ReviewAuditEventType.ADJUDICATED.value,
            details={
                "action": "ADJUDICATE_SPLIT_DECISION",
                "final_decision": decision.value,
                "adjudicator_id": adjudicator.id,
                "comments": comments.strip()[:200]
            },
            review_case_id=case.id
        )

        db.commit()
        return case

    # -------------------------------------------------------------------------
    # 6. Structured Revision Workflow & Resubmission Rounds
    # -------------------------------------------------------------------------

    @classmethod
    def submit_revision_response(
        cls,
        db: Session,
        case_id: str,
        user: User,
        written_response: str,
        new_evidence_file_ids: List[str]
    ) -> ReviewCase:
        """
        NGO submits corrected evidence and responses to revision requests.
        Preserves prior rounds and decisions intact; creates round N+1 for independent re-review.
        """
        prior_case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not prior_case:
            raise NotFoundException("Review case not found.")

        if prior_case.status != ReviewCaseStatus.REVISION_REQUIRED.value:
            raise BadRequestException(f"Cannot submit revision response for case in status '{prior_case.status}'. Must be REVISION_REQUIRED.")

        # Check revision deadline
        if prior_case.revision_deadline and datetime.utcnow() > prior_case.revision_deadline and user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("The revision submission deadline for this review case has expired.")

        # Validate NGO submitter authorization
        if user.role != PlatformRole.PLATFORM_ADMIN:
            membership = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == prior_case.organization_id,
                        OrganizationMembership.user_id == user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            if not membership:
                raise ForbiddenException("You are not authorized to submit revisions for this organization.")

        if not written_response or len(written_response.strip()) < 10:
            raise BadRequestException("A written explanation addressing the revision requests is required.")

        if not new_evidence_file_ids:
            raise BadRequestException("At least one replacement or supporting evidence file must be provided.")

        # Validate evidence files
        new_ev_files = db.execute(
            select(EvidenceFile).where(
                and_(EvidenceFile.id.in_(new_evidence_file_ids), EvidenceFile.is_archived == False)
            )
        ).scalars().all()
        if len(new_ev_files) != len(new_evidence_file_ids):
            raise NotFoundException("One or more specified evidence files were not found or are archived.")

        # Compute new frozen snapshot digest
        digest_components = []
        for ev in new_ev_files:
            digest_components.append(f"{ev.id}:{ev.version_number}:{ev.file_hash}")
        new_digest = hashlib.sha256("|".join(sorted(digest_components)).encode("utf-8")).hexdigest()

        # Create new submission round case
        new_round = prior_case.submission_round + 1
        new_case = ReviewCase(
            id=str(uuid.uuid4()),
            milestone_id=prior_case.milestone_id,
            project_id=prior_case.project_id,
            organization_id=prior_case.organization_id,
            submission_round=new_round,
            submitted_by_id=user.id,
            status=ReviewCaseStatus.SUBMITTED.value,
            submission_notes=f"Revision Response (Round {new_round}): {written_response.strip()}",
            snapshot_digest=new_digest,
            snapshot_data_json={
                "prior_case_id": prior_case.id,
                "submission_round": new_round,
                "written_response": written_response.strip(),
                "evidence_count": len(new_ev_files),
                "digest": new_digest
            },
            review_policy_version=settings.REVIEW_POLICY_VERSION,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_case)
        db.flush()

        for ev in new_ev_files:
            rce = ReviewCaseEvidence(
                id=str(uuid.uuid4()),
                review_case_id=new_case.id,
                evidence_file_id=ev.id,
                evidence_version=ev.version_number,
                evidence_hash=ev.file_hash,
                evidence_role="REVISED",
                snapshot_timestamp=datetime.utcnow()
            )
            db.add(rce)

        # Mark prior case superseded / closed
        prior_case.status = ReviewCaseStatus.CLOSED.value

        # Update milestone status back to VERIFICATION_PENDING
        milestone = db.execute(select(Milestone).where(Milestone.id == prior_case.milestone_id)).scalar_one_or_none()
        if milestone:
            milestone.status = MilestoneStatus.VERIFICATION_PENDING

        cls._append_audit_event(
            db=db,
            milestone_id=prior_case.milestone_id,
            actor_id=user.id,
            event_type=ReviewAuditEventType.NGO_RESUBMITTED.value,
            details={
                "action": "SUBMIT_REVISION_RESPONSE",
                "prior_case_id": prior_case.id,
                "new_case_id": new_case.id,
                "submission_round": new_round,
                "evidence_count": len(new_ev_files)
            },
            review_case_id=new_case.id
        )

        db.commit()
        return new_case

    # -------------------------------------------------------------------------
    # 7. Formal Dispute & Appeal Governance
    # -------------------------------------------------------------------------

    @classmethod
    def raise_dispute(
        cls,
        db: Session,
        case_id: str,
        user: User,
        category: DisputeCategory,
        reason: str,
        supporting_evidence: Optional[Dict[str, Any]] = None
    ) -> ReviewDispute:
        """
        NGO raises a formal appeal against a final adverse decision (e.g. REJECTED).
        Enforces appeal window and single-appeal limits.
        """
        case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException("Review case not found.")

        # Eligibility check 1: Case must be REJECTED
        if case.status != ReviewCaseStatus.REJECTED.value and case.final_decision != ReviewDecision.REJECT.value:
            raise BadRequestException("Disputes may only be formally raised against rejected milestone submissions.")

        # Eligibility check 2: Appeal window must not be expired
        if case.appeal_deadline and datetime.utcnow() > case.appeal_deadline and user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("The formal appeal deadline for this review case has expired.")

        # Eligibility check 3: User must belong to the organization
        if user.role != PlatformRole.PLATFORM_ADMIN:
            membership = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == case.organization_id,
                        OrganizationMembership.user_id == user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            if not membership or membership.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
                raise ForbiddenException("Only organization administrators may file formal milestone disputes.")

        # Eligibility check 4: Max dispute limit
        existing_disputes_count = db.execute(
            select(func.count(ReviewDispute.id)).where(ReviewDispute.review_case_id == case.id)
        ).scalar() or 0
        if existing_disputes_count >= settings.REVIEW_MAX_APPEALS_PER_CASE:
            raise ConflictException(f"Maximum allowed formal appeals ({settings.REVIEW_MAX_APPEALS_PER_CASE}) already filed for this case.")

        if not reason or len(reason.strip()) < 15:
            raise BadRequestException("A substantiated dispute statement (minimum 15 characters) is required.")

        dispute = ReviewDispute(
            id=str(uuid.uuid4()),
            review_case_id=case.id,
            milestone_id=case.milestone_id,
            project_id=case.project_id,
            organization_id=case.organization_id,
            original_decision_record_id=None,
            raised_by_id=user.id,
            status=DisputeStatus.OPEN.value,
            dispute_category=category.value,
            dispute_reason=reason.strip(),
            supporting_evidence_json=supporting_evidence,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(dispute)

        case.status = ReviewCaseStatus.DISPUTED.value

        cls._append_audit_event(
            db=db,
            milestone_id=case.milestone_id,
            actor_id=user.id,
            event_type=ReviewAuditEventType.DISPUTE_RAISED.value,
            details={
                "dispute_id": dispute.id,
                "case_id": case.id,
                "category": category.value,
                "reason": reason.strip()[:200]
            },
            review_case_id=case.id
        )

        db.commit()
        return dispute

    @classmethod
    def add_dispute_message(
        cls,
        db: Session,
        dispute_id: str,
        user: User,
        message: str,
        attachments: Optional[Dict[str, Any]] = None
    ) -> ReviewDisputeMessage:
        """Appends an auditable communication message to a formal dispute thread."""
        dispute = db.execute(select(ReviewDispute).where(ReviewDispute.id == dispute_id)).scalar_one_or_none()
        if not dispute:
            raise NotFoundException("Dispute not found.")

        # Authorization: Must be admin, adjudicator, or org member
        is_admin = user.role == PlatformRole.PLATFORM_ADMIN
        is_adjudicator = dispute.assigned_adjudicator_id == user.id
        is_org_member = False
        if not is_admin and not is_adjudicator:
            mem = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == dispute.organization_id,
                        OrganizationMembership.user_id == user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            is_org_member = bool(mem)

        if not (is_admin or is_adjudicator or is_org_member):
            raise ForbiddenException("You are not an authorized participant in this dispute.")

        if not message or len(message.strip()) < 3:
            raise BadRequestException("Dispute message cannot be empty.")

        sender_role = "ADMIN" if is_admin else ("ADJUDICATOR" if is_adjudicator else "NGO_APPELLANT")

        msg = ReviewDisputeMessage(
            id=str(uuid.uuid4()),
            dispute_id=dispute.id,
            sender_id=user.id,
            sender_role=sender_role,
            message=message.strip(),
            attachments_json=attachments,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(msg)

        cls._append_audit_event(
            db=db,
            milestone_id=dispute.milestone_id,
            actor_id=user.id,
            event_type=ReviewAuditEventType.DISPUTE_MESSAGE_ADDED.value,
            details={"dispute_id": dispute.id, "sender_role": sender_role},
            review_case_id=dispute.review_case_id
        )

        db.commit()
        return msg

    @classmethod
    def resolve_dispute(
        cls,
        db: Session,
        dispute_id: str,
        adjudicator: User,
        resolution: DisputeResolution,
        explanation: str
    ) -> ReviewDispute:
        """
        Independent adjudicator issues final binding resolution on a milestone appeal.
        Preserves original records; if overturned or re-review ordered, reopens milestone.
        """
        if adjudicator.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only platform administrators may resolve formal milestone disputes.")

        dispute = db.execute(select(ReviewDispute).where(ReviewDispute.id == dispute_id)).scalar_one_or_none()
        if not dispute:
            raise NotFoundException("Dispute not found.")

        case = db.execute(select(ReviewCase).where(ReviewCase.id == dispute.review_case_id)).scalar_one_or_none()
        milestone = db.execute(select(Milestone).where(Milestone.id == dispute.milestone_id)).scalar_one_or_none()

        # Adjudicator COI check: cannot be one of the original reviewers
        prior_reviewers = db.execute(
            select(ReviewDecisionRecord.reviewer_id).where(ReviewDecisionRecord.review_case_id == dispute.review_case_id)
        ).scalars().all()
        if adjudicator.id in prior_reviewers:
            raise ForbiddenException("Conflict of interest: Cannot adjudicate an appeal on a case you originally reviewed.")

        if not explanation or len(explanation.strip()) < 15:
            raise BadRequestException("A thorough resolution explanation (minimum 15 characters) is required.")

        dispute.assigned_adjudicator_id = adjudicator.id
        dispute.resolution = resolution.value
        dispute.resolution_explanation = explanation.strip()
        dispute.resolved_at = datetime.utcnow()
        dispute.status = DisputeStatus.RESOLVED.value

        # Effect on case and milestone
        if resolution in (DisputeResolution.OVERTURN_ORIGINAL_DECISION, DisputeResolution.ORDER_RE_REVIEW):
            if case:
                case.status = ReviewCaseStatus.RESOLVED.value
            if milestone:
                # Reopen milestone for verification / review
                milestone.status = MilestoneStatus.VERIFICATION_PENDING

        elif resolution == DisputeResolution.REQUEST_NEW_EVIDENCE:
            if case:
                case.status = ReviewCaseStatus.REVISION_REQUIRED.value
                case.revision_deadline = datetime.utcnow() + timedelta(hours=settings.REVIEW_REVISION_DUE_HOURS)
            if milestone:
                milestone.status = MilestoneStatus.REVISION_REQUIRED

        else:  # UPHOLD_ORIGINAL_DECISION or DISMISS_DISPUTE
            if case:
                case.status = ReviewCaseStatus.CLOSED.value

        cls._append_audit_event(
            db=db,
            milestone_id=dispute.milestone_id,
            actor_id=adjudicator.id,
            event_type=ReviewAuditEventType.DISPUTE_RESOLVED.value,
            details={
                "dispute_id": dispute.id,
                "case_id": dispute.review_case_id,
                "resolution": resolution.value,
                "explanation": explanation.strip()[:200]
            },
            review_case_id=dispute.review_case_id
        )

        db.commit()
        return dispute

    # -------------------------------------------------------------------------
    # 8. Queue Queries & Blind Review Filtering
    # -------------------------------------------------------------------------

    @classmethod
    def get_reviewer_queue(
        cls,
        db: Session,
        reviewer: User,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves review tasks assigned to the current reviewer,
        enriching each with milestone, project, and organization metadata.
        """
        query = select(ReviewTask).where(ReviewTask.assigned_reviewer_id == reviewer.id)
        if status_filter:
            query = query.where(ReviewTask.assignment_state == status_filter)

        tasks = db.execute(query.order_by(ReviewTask.assigned_at.desc())).scalars().all()
        results = []
        for t in tasks:
            case = db.execute(select(ReviewCase).where(ReviewCase.id == t.review_case_id)).scalar_one_or_none()
            milestone = db.execute(select(Milestone).where(Milestone.id == t.milestone_id)).scalar_one_or_none()
            project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none() if milestone else None
            org = db.execute(select(Organization).where(Organization.id == project.organization_id)).scalar_one_or_none() if project else None

            results.append({
                "task_id": t.id,
                "case_id": t.review_case_id,
                "slot": t.reviewer_slot,
                "assignment_state": t.assignment_state,
                "due_at": t.due_at,
                "milestone_id": t.milestone_id,
                "milestone_title": milestone.title if milestone else "Unknown",
                "project_title": project.title if project else "Unknown",
                "organization_name": org.legal_name if org else "Unknown",
                "submission_round": case.submission_round if case else 1,
                "case_status": case.status if case else "UNKNOWN"
            })
        return results

    @classmethod
    def get_case_detail_for_reviewer(
        cls,
        db: Session,
        case_id: str,
        user: User
    ) -> Dict[str, Any]:
        """
        Retrieves complete review case detail.
        CRITICAL: Implements blind review protection:
        If user is assigned as Reviewer 2 and has not yet submitted a decision,
        Reviewer 1's decision and comments are strictly hidden to prevent anchoring bias.
        """
        case = db.execute(select(ReviewCase).where(ReviewCase.id == case_id)).scalar_one_or_none()
        if not case:
            raise NotFoundException("Review case not found.")

        milestone = db.execute(select(Milestone).where(Milestone.id == case.milestone_id)).scalar_one_or_none()
        project = db.execute(select(Project).where(Project.id == case.project_id)).scalar_one_or_none()
        org = db.execute(select(Organization).where(Organization.id == case.organization_id)).scalar_one_or_none()

        # Check access permission: Reviewer assigned, Platform Admin, or NGO member of org
        is_admin = user.role == PlatformRole.PLATFORM_ADMIN
        user_tasks = db.execute(
            select(ReviewTask).where(
                and_(ReviewTask.review_case_id == case.id, ReviewTask.assigned_reviewer_id == user.id)
            )
        ).scalars().all()

        is_assigned_reviewer = bool(user_tasks)
        is_org_member = False
        if not is_admin and not is_assigned_reviewer:
            mem = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == case.organization_id,
                        OrganizationMembership.user_id == user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            is_org_member = bool(mem)

        if not (is_admin or is_assigned_reviewer or is_org_member):
            raise ForbiddenException("You do not have permission to view this review case.")

        # Determine if blind review applies
        user_decided = any(t.assignment_state == ReviewAssignmentStatus.DECIDED.value for t in user_tasks)
        blind_review_active = is_assigned_reviewer and not user_decided and not is_admin

        # Fetch decisions
        decisions_query = select(ReviewDecisionRecord).where(
            and_(
                ReviewDecisionRecord.review_case_id == case.id,
                ReviewDecisionRecord.submission_round == case.submission_round
            )
        )
        decisions = db.execute(decisions_query).scalars().all()

        decision_payload = []
        for d in decisions:
            if blind_review_active and d.reviewer_id != user.id:
                # Mask other reviewer's decision to eliminate anchoring bias
                decision_payload.append({
                    "decision_id": d.id,
                    "reviewer_id": "REDACTED_BLIND_REVIEW",
                    "decision": "SUBMITTED_HIDDEN",
                    "comments": "Decision and rationale masked pending your independent review submission.",
                    "decided_at": d.decided_at
                })
            else:
                decision_payload.append({
                    "decision_id": d.id,
                    "reviewer_id": d.reviewer_id,
                    "decision": d.decision.value,
                    "comments": d.comments,
                    "decided_at": d.decided_at,
                    "rejection_category": d.rejection_reason_category,
                    "structured_revision_data": d.structured_revision_data_json
                })

        # Fetch evidence package
        evidence_items = []
        for rce in case.evidence_items:
            ev = rce.evidence_file
            report = rce.verification_report
            evidence_items.append({
                "evidence_id": rce.evidence_file_id,
                "original_filename": ev.original_filename if ev else "Unknown",
                "mime_type": ev.detected_mime_type if ev else "application/octet-stream",
                "version": rce.evidence_version,
                "file_hash": rce.evidence_hash,
                "verification_report": {
                    "report_id": report.id,
                    "overall_assessment": report.overall_assessment,
                    "confidence_score": report.confidence_score,
                    "calibration_status": report.calibration_status,
                    "summary_notes": report.summary_notes,
                    "checks": [
                        {
                            "check_type": c.check_type,
                            "status": c.status,
                            "passed": c.passed,
                            "message": c.message
                        } for c in (report.checks if report else [])
                    ]
                } if report else None
            })

        return {
            "case_id": case.id,
            "milestone_id": case.milestone_id,
            "milestone_title": milestone.title if milestone else "Unknown",
            "milestone_description": milestone.description if milestone else "",
            "project_title": project.title if project else "Unknown",
            "organization_name": org.legal_name if org else "Unknown",
            "submission_round": case.submission_round,
            "status": case.status,
            "final_decision": case.final_decision if not blind_review_active else None,
            "reconciliation_result": case.reconciliation_result if not blind_review_active else None,
            "submission_notes": case.submission_notes,
            "snapshot_digest": case.snapshot_digest,
            "evidence_package": evidence_items,
            "decisions": decision_payload,
            "blind_review_active": blind_review_active,
            "revision_deadline": case.revision_deadline,
            "appeal_deadline": case.appeal_deadline,
            "created_at": case.created_at
        }

    @classmethod
    def get_review_audit_trail(cls, db: Session, milestone_id: str) -> Dict[str, Any]:
        """
        Retrieves the complete cryptographic audit chain for a milestone's reviews,
        verifying unbroken SHA-256 hash continuity.
        """
        events = db.execute(
            select(ReviewAuditEvent)
            .where(ReviewAuditEvent.milestone_id == milestone_id)
            .order_by(ReviewAuditEvent.created_at.asc())
        ).scalars().all()

        is_intact = True
        prev_hash = "0" * 64

        for ev in events:
            expected = compute_review_audit_hash(
                previous_hash=prev_hash,
                milestone_id=ev.milestone_id,
                actor_id=ev.actor_id,
                event_type=ev.event_type,
                details=ev.details_json or {}
            )
            if ev.current_event_hash != expected or ev.previous_event_hash != prev_hash:
                is_intact = False
                break
            prev_hash = ev.current_event_hash

        return {
            "milestone_id": milestone_id,
            "total_events": len(events),
            "is_chain_intact": is_intact,
            "events": [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "actor_id": e.actor_id,
                    "details": e.details_json,
                    "current_hash": e.current_event_hash,
                    "created_at": e.created_at
                } for e in events
            ]
        }
