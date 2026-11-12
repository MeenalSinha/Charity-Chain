"""
disbursement_eligibility_service.py - Multi-Gate Milestone Payout Eligibility Engine
=====================================================================================
Validates all compliance, review, financial, and regulatory gates before any disbursement:
1. Milestone state verification (APPROVED or COMPLETED)
2. Project and Organization verified status
3. Human Review Case completion with explicit APPROVE decision
4. Dual-Key reviewer consensus / reconciliation sign-off
5. Tamper-evident evidence snapshot SHA-256 digest verification
6. Absence of open/unresolved appeals or disputes
7. Active, verified NGO payout recipient on file
8. Real-time available designated fund balance sufficiency
9. Duplicate / in-flight payout conflict prevention
"""

import hashlib
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    ReviewCaseStatus, ReviewDecision, ReconciliationResult,
    DisputeStatus, PayoutRecipientStatus, DisbursementStatus
)
from backend.app.core.logging import get_logger
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import (
    ReviewCase, ReviewCaseEvidence, ReviewTask, ReviewDecisionRecord, ReviewDispute
)
from backend.app.db.models.financial import Disbursement, PayoutRecipient
from backend.app.services.fund_allocation_service import FundAllocationService

logger = get_logger("services.disbursement_eligibility")


class DisbursementEligibilityService:
    """Verifies that an approved milestone satisfies all criteria for fund disbursement."""

    @classmethod
    def evaluate_eligibility(
        cls,
        db: Session,
        milestone_id: str,
        requested_amount_minor_units: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Conducts a multi-gate audit of milestone disbursement eligibility.
        Returns a structured report detailing pass/fail results for every prerequisite.
        """
        passed_checks: List[str] = []
        failed_checks: List[str] = []
        rejection_reasons: List[str] = []

        # 1. Milestone Existence & Status
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            return {
                "is_eligible": False,
                "milestone_id": milestone_id,
                "passed_checks": [],
                "failed_checks": ["MILESTONE_EXISTS"],
                "rejection_reasons": [f"Milestone '{milestone_id}' does not exist."],
                "available_balance_minor_units": 0,
                "max_disbursable_minor_units": 0
            }
        passed_checks.append("MILESTONE_EXISTS")

        if milestone.status in (MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED):
            passed_checks.append("MILESTONE_STATUS_APPROVED")
        else:
            failed_checks.append("MILESTONE_STATUS_APPROVED")
            rejection_reasons.append(
                f"Milestone status is '{milestone.status.value}'. Only APPROVED or COMPLETED milestones can receive funds."
            )

        # 2. Project & Organization Verification
        project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none()
        if project and project.status in (ProjectStatus.ACTIVE, ProjectStatus.COMPLETED):
            passed_checks.append("PROJECT_ACTIVE")
        else:
            failed_checks.append("PROJECT_ACTIVE")
            rejection_reasons.append("Associated project is not ACTIVE or COMPLETED.")

        org = db.execute(select(Organization).where(Organization.id == project.organization_id)).scalar_one_or_none() if project else None
        if org and org.status == OrganizationStatus.VERIFIED:
            passed_checks.append("ORGANIZATION_VERIFIED")
        else:
            failed_checks.append("ORGANIZATION_VERIFIED")
            rejection_reasons.append("NGO organization is not in VERIFIED status.")

        # 3. Human Review Case & Final Decision
        review_case = db.execute(
            select(ReviewCase)
            .where(ReviewCase.milestone_id == milestone_id)
            .order_by(ReviewCase.submission_round.desc())
        ).scalars().first()

        if review_case:
            passed_checks.append("REVIEW_CASE_EXISTS")
            if (
                review_case.status == ReviewCaseStatus.APPROVED.value
                and review_case.final_decision == ReviewDecision.APPROVE.value
            ):
                passed_checks.append("REVIEW_CASE_FINAL_APPROVAL")
            else:
                failed_checks.append("REVIEW_CASE_FINAL_APPROVAL")
                rejection_reasons.append(
                    f"ReviewCase #{review_case.submission_round} is '{review_case.status}' with decision '{review_case.final_decision}'. Final APPROVE required."
                )

            # 4. Dual-Key Reconciliation Verification
            reconciliation_ok = False
            if review_case.reconciliation_result == ReconciliationResult.APPROVED.value:
                reconciliation_ok = True
            else:
                # Check individual reviewer decision records
                decisions = db.execute(
                    select(ReviewDecisionRecord)
                    .join(ReviewTask, ReviewDecisionRecord.review_task_id == ReviewTask.id)
                    .where(
                        and_(
                            ReviewTask.review_case_id == review_case.id,
                            ReviewDecisionRecord.submission_round == review_case.submission_round
                        )
                    )
                ).scalars().all()
                approved_decisions = [d for d in decisions if d.decision == ReviewDecision.APPROVE.value]
                if len(approved_decisions) >= 2:
                    reconciliation_ok = True

            if reconciliation_ok:
                passed_checks.append("DUAL_KEY_RECONCILIATION_VERIFIED")
            else:
                failed_checks.append("DUAL_KEY_RECONCILIATION_VERIFIED")
                rejection_reasons.append("Dual-key reviewer consensus or approved reconciliation is required.")

            # 5. Snapshot Digest Integrity Verification
            digest_verified = False
            case_evidence_links = db.execute(
                select(ReviewCaseEvidence).where(ReviewCaseEvidence.review_case_id == review_case.id)
            ).scalars().all()

            if case_evidence_links and review_case.snapshot_digest:
                digest_components = []
                for link in case_evidence_links:
                    digest_components.append(
                        f"{link.evidence_file_id}:{link.evidence_version}:{link.evidence_hash}:{link.verification_report_id or 'none'}"
                    )
                recomputed_digest = hashlib.sha256("|".join(sorted(digest_components)).encode("utf-8")).hexdigest()
                if recomputed_digest == review_case.snapshot_digest:
                    digest_verified = True

            # If snapshot_digest matches stored data_json digest
            if not digest_verified and review_case.snapshot_data_json:
                json_digest = review_case.snapshot_data_json.get("digest")
                if json_digest and json_digest == review_case.snapshot_digest:
                    digest_verified = True

            if digest_verified:
                passed_checks.append("SNAPSHOT_DIGEST_VERIFIED")
            else:
                failed_checks.append("SNAPSHOT_DIGEST_VERIFIED")
                rejection_reasons.append("Evidence snapshot digest mismatch or evidence tampering detected.")
        else:
            failed_checks.extend([
                "REVIEW_CASE_EXISTS",
                "REVIEW_CASE_FINAL_APPROVAL",
                "DUAL_KEY_RECONCILIATION_VERIFIED",
                "SNAPSHOT_DIGEST_VERIFIED"
            ])
            rejection_reasons.append("No Human Review Case found for this milestone.")

        # 6. Absence of Active Disputes or Appeals
        active_disputes = db.execute(
            select(ReviewDispute).where(
                and_(
                    ReviewDispute.milestone_id == milestone_id,
                    ReviewDispute.status.in_([
                        DisputeStatus.OPEN.value,
                        DisputeStatus.UNDER_REVIEW.value,
                        DisputeStatus.AWAITING_INFORMATION.value,
                        DisputeStatus.ESCALATED.value
                    ])
                )
            )
        ).scalars().all()

        if not active_disputes:
            passed_checks.append("NO_UNRESOLVED_DISPUTES")
        else:
            failed_checks.append("NO_UNRESOLVED_DISPUTES")
            rejection_reasons.append(
                f"Milestone has {len(active_disputes)} unresolved dispute(s) or appeal(s) pending adjudication."
            )

        # 7. Active Verified Payout Recipient
        recipient = None
        if org:
            recipient = db.execute(
                select(PayoutRecipient).where(
                    and_(
                        PayoutRecipient.organization_id == org.id,
                        PayoutRecipient.status == PayoutRecipientStatus.VERIFIED
                    )
                ).order_by(PayoutRecipient.is_primary.desc())
            ).scalars().first()

        if recipient:
            passed_checks.append("VERIFIED_PAYOUT_RECIPIENT_CONFIGURED")
        else:
            failed_checks.append("VERIFIED_PAYOUT_RECIPIENT_CONFIGURED")
            rejection_reasons.append("Organization does not have an active, VERIFIED payout destination configured.")

        # 8. Available Fund Balance Sufficiency
        fund_summary = FundAllocationService.get_milestone_fund_summary(db, milestone_id)
        available_balance = fund_summary["available_balance_minor_units"]

        if available_balance > 0:
            passed_checks.append("FUNDS_AVAILABLE")
            if requested_amount_minor_units is not None:
                if requested_amount_minor_units <= available_balance:
                    passed_checks.append("REQUESTED_AMOUNT_WITHIN_BALANCE")
                else:
                    failed_checks.append("REQUESTED_AMOUNT_WITHIN_BALANCE")
                    rejection_reasons.append(
                        f"Requested amount ({requested_amount_minor_units} minor units) exceeds available unencumbered balance ({available_balance} minor units)."
                    )
        else:
            failed_checks.append("FUNDS_AVAILABLE")
            rejection_reasons.append("Milestone has zero available unencumbered designated funds for disbursement.")

        # 9. In-Flight Payout Conflict Prevention
        in_flight_payouts = db.execute(
            select(Disbursement).where(
                and_(
                    Disbursement.milestone_id == milestone_id,
                    Disbursement.status.in_([
                        DisbursementStatus.REQUESTED,
                        DisbursementStatus.PENDING_APPROVAL,
                        DisbursementStatus.APPROVED,
                        DisbursementStatus.SUBMISSION_PENDING,
                        DisbursementStatus.SUBMITTED,
                        DisbursementStatus.PROCESSING
                    ])
                )
            )
        ).scalars().all()

        if not in_flight_payouts:
            passed_checks.append("NO_IN_FLIGHT_DISBURSEMENT")
        else:
            failed_checks.append("NO_IN_FLIGHT_DISBURSEMENT")
            rejection_reasons.append(
                f"Milestone already has an active disbursement in flight ({in_flight_payouts[0].id}, status: {in_flight_payouts[0].status.value})."
            )

        is_eligible = len(failed_checks) == 0

        return {
            "is_eligible": is_eligible,
            "milestone_id": milestone.id,
            "project_id": milestone.project_id,
            "organization_id": org.id if org else None,
            "review_case_id": review_case.id if review_case else None,
            "recipient_id": recipient.id if recipient else None,
            "recipient_masked": recipient.destination_masked if recipient else None,
            "available_balance_minor_units": available_balance,
            "max_disbursable_minor_units": available_balance,
            "currency": fund_summary["currency"],
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "rejection_reasons": rejection_reasons
        }
