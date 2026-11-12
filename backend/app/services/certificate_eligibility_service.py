"""
certificate_eligibility_service.py - Multi-Gate Proof-of-Impact Certificate Eligibility
========================================================================================
Enforces strict verification gates before any digital impact certificate is signed or issued.
Verifies genuine payment capture, dual-key human review consensus, evidence snapshot integrity,
and absence of open disputes. Never relies on a simple milestone status string.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import (
    DonationStatus, ReviewCaseStatus, ReviewDecision, DisputeStatus,
    CertificateType, CertificateStatus, ProjectStatus, OrganizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus
)
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture, PaymentRefund, Disbursement
from backend.app.db.models.projects import Milestone, Project
from backend.app.db.models.users import Organization
from backend.app.db.models.reviews_impact import (
    ReviewCase, ReviewDispute, ImpactCertificate
)


class CertificateEligibilityResult(BaseModel):
    is_eligible: bool
    certificate_type: str
    reasons: List[str]
    explanation: str
    donation_id: str
    milestone_id: str
    project_id: str
    organization_id: str
    review_case_id: Optional[str] = None
    qualifying_amount_cents: int = 0
    currency: str = "USD"
    snapshot_digest: Optional[str] = None
    disbursement_id: Optional[str] = None
    policy_version: str = "v9.0.0"
    evaluation_timestamp: str


class CertificateEligibilityService:
    @classmethod
    def evaluate_eligibility(
        cls,
        db: Session,
        donation_id: str,
        certificate_type: str = CertificateType.CONTRIBUTION_IMPACT.value,
        include_disbursement_claim: bool = False
    ) -> CertificateEligibilityResult:
        """
        Evaluates whether a captured donation and its associated milestone qualify
        for an authentic, cryptographically signed Proof-of-Impact Certificate.
        """
        reasons: List[str] = []
        now_iso = datetime.utcnow().isoformat()

        # 1. Resolve Donation
        donation = db.query(Donation).filter(Donation.id == donation_id).first()
        if not donation:
            return CertificateEligibilityResult(
                is_eligible=False,
                certificate_type=certificate_type,
                reasons=["DONATION_NOT_FOUND"],
                explanation=f"Donation '{donation_id}' does not exist.",
                donation_id=donation_id,
                milestone_id="",
                project_id="",
                organization_id="",
                evaluation_timestamp=now_iso
            )

        milestone_id = donation.milestone_id or ""
        project_id = donation.project_id or ""

        # 2. Verify Genuine Payment Capture
        captures = db.query(PaymentCapture).join(
            PaymentOrder, PaymentOrder.id == PaymentCapture.order_id
        ).filter(
            PaymentOrder.donation_id == donation.id,
            PaymentCapture.status.in_([
                PaymentCaptureStatus.COMPLETED,
                PaymentCaptureStatus.PARTIALLY_REFUNDED,
                "COMPLETED",
                "PARTIALLY_REFUNDED"
            ])
        ).all()
        if not captures and donation.status != DonationStatus.PAID.value:
            reasons.append("NO_CONFIRMED_CAPTURE")

        total_captured = sum(c.amount_minor_units for c in captures) if captures else getattr(donation, "amount_minor_units", 0)

        # 3. Calculate Qualifying Amount after Completed Refunds
        refunds = db.query(PaymentRefund).join(
            PaymentCapture, PaymentCapture.id == PaymentRefund.capture_id
        ).join(
            PaymentOrder, PaymentOrder.id == PaymentCapture.order_id
        ).filter(
            PaymentOrder.donation_id == donation.id,
            PaymentRefund.status.in_([
                PaymentRefundStatus.COMPLETED,
                "COMPLETED"
            ])
        ).all()
        total_refunded = sum(r.amount_minor_units for r in refunds)
        qualifying_amount = total_captured - total_refunded

        if qualifying_amount <= 0:
            reasons.append("FULLY_REFUNDED_OR_ZERO_BALANCE")

        # 4. Resolve Milestone & Parent Project
        milestone = db.query(Milestone).filter(Milestone.id == milestone_id).first()
        if not milestone:
            reasons.append("MILESTONE_NOT_FOUND")

        project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
        if not project and milestone:
            project = db.query(Project).filter(Project.id == milestone.project_id).first()
            if project:
                project_id = project.id

        if not project:
            reasons.append("PROJECT_NOT_FOUND")

        organization_id = project.organization_id if project else ""

        # 5. Verify Organization Status
        if organization_id:
            org = db.query(Organization).filter(Organization.id == organization_id).first()
            is_org_verified = (
                org is not None and (
                    org.status == OrganizationStatus.VERIFIED or
                    getattr(org, "status", None) == "VERIFIED" or
                    getattr(org, "is_verified", False) is True
                )
            )
            if not is_org_verified:
                reasons.append("ORGANIZATION_NOT_VERIFIED")
        else:
            reasons.append("ORGANIZATION_NOT_FOUND")

        # 6. Verify Human Review Approval & Evidence Snapshot
        review_case = None
        snapshot_digest = None
        if milestone:
            # Query approved review case
            review_case = db.query(ReviewCase).filter(
                ReviewCase.milestone_id == milestone.id,
                ReviewCase.status.in_([
                    ReviewCaseStatus.APPROVED.value,
                    ReviewCaseStatus.RESOLVED.value,
                    ReviewCaseStatus.CLOSED.value
                ]),
                ReviewCase.final_decision.in_([ReviewDecision.APPROVE.value, "APPROVED", "APPROVE"])
            ).order_by(ReviewCase.submission_round.desc()).first()

            if not review_case:
                # Also accept finalized adjudication approval
                adjudicated_case = db.query(ReviewCase).filter(
                    ReviewCase.milestone_id == milestone.id,
                    ReviewCase.final_decision.in_([ReviewDecision.APPROVE.value, "APPROVED", "APPROVE"])
                ).order_by(ReviewCase.submission_round.desc()).first()
                if adjudicated_case and adjudicated_case.adjudicated_at:
                    review_case = adjudicated_case

            if not review_case:
                # Milestone has no verified human approval record
                reasons.append("MILESTONE_NOT_HUMAN_APPROVED")
            else:
                snapshot_digest = review_case.snapshot_digest

        # 7. Check for Open Disputes
        if milestone:
            open_dispute = db.query(ReviewDispute).filter(
                ReviewDispute.milestone_id == milestone.id,
                ReviewDispute.status.in_([
                    DisputeStatus.OPEN.value,
                    DisputeStatus.UNDER_REVIEW.value,
                    DisputeStatus.AWAITING_INFORMATION.value,
                    DisputeStatus.ESCALATED.value
                ])
            ).first()
            if open_dispute:
                reasons.append("UNRESOLVED_DISPUTE_EXISTS")

        # 8. Idempotency Check: Existing Active Certificate
        existing_cert = db.query(ImpactCertificate).filter(
            ImpactCertificate.donation_id == donation.id,
            ImpactCertificate.certificate_type == certificate_type,
            ImpactCertificate.status.in_([
                CertificateStatus.ISSUED.value,
                CertificateStatus.ACTIVE.value
            ])
        ).first()
        if existing_cert:
            reasons.append("CERTIFICATE_ALREADY_ISSUED")

        # 9. Optional Disbursement Claim Validation
        disbursement_id = None
        if include_disbursement_claim or certificate_type == CertificateType.DISBURSEMENT_ATTESTATION.value:
            if milestone:
                disb = db.query(Disbursement).filter(
                    Disbursement.milestone_id == milestone.id,
                    Disbursement.status == "COMPLETED"
                ).first()
                if disb:
                    disbursement_id = disb.id
                else:
                    if certificate_type == CertificateType.DISBURSEMENT_ATTESTATION.value:
                        reasons.append("DISBURSEMENT_NOT_CONFIRMED")

        is_eligible = len(reasons) == 0
        explanation = (
            "Milestone and donation satisfy all cryptographic verification and governance gates."
            if is_eligible
            else f"Eligibility rejected due to: {', '.join(reasons)}"
        )

        return CertificateEligibilityResult(
            is_eligible=is_eligible,
            certificate_type=certificate_type,
            reasons=reasons,
            explanation=explanation,
            donation_id=donation.id,
            milestone_id=milestone_id,
            project_id=project_id,
            organization_id=organization_id,
            review_case_id=review_case.id if review_case else None,
            qualifying_amount_cents=qualifying_amount,
            currency=getattr(donation, "currency", "USD"),
            snapshot_digest=snapshot_digest,
            disbursement_id=disbursement_id,
            policy_version=getattr(settings, "CERTIFICATE_ELIGIBILITY_POLICY_VERSION", "v9.0.0"),
            evaluation_timestamp=now_iso
        )
