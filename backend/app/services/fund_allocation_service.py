"""
fund_allocation_service.py - Real-Time Fund Balance & Allocation Service
=======================================================================
Calculates strictly segregated fund balances for projects and milestones:
- Gross captured contributions
- Completed donor refunds
- Net designated funds
- Committed payout liabilities
- Succeeded payouts
- Net available balance for new disbursements
- Strict integer arithmetic in minor units (cents) with zero floating-point error.
"""

from typing import Dict, Any, Optional
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    DonationStatus, PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus
)
from backend.app.core.exceptions import NotFoundException
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentCapture, PaymentRefund, Disbursement
)


class FundAllocationService:
    """Computes exact, unencumbered fund balances available for milestone disbursements."""

    @staticmethod
    def get_milestone_fund_summary(
        db: Session,
        milestone_id: str
    ) -> Dict[str, Any]:
        """
        Calculates exact financial ledger and allocation metrics for a single milestone.
        """
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException(f"Milestone '{milestone_id}' not found.")

        project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none()
        currency = milestone.currency or (project.currency if project else "USD")

        # 1. Gross Captured from completed captures
        gross_captured = db.execute(
            select(func.coalesce(func.sum(PaymentCapture.amount_minor_units), 0))
            .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
            .join(Donation, PaymentOrder.donation_id == Donation.id)
            .where(
                and_(
                    Donation.milestone_id == milestone_id,
                    PaymentCapture.status.in_([
                        PaymentCaptureStatus.COMPLETED,
                        PaymentCaptureStatus.PARTIALLY_REFUNDED,
                        PaymentCaptureStatus.REFUNDED
                    ])
                )
            )
        ).scalar() or 0

        # 2. Completed Refunds
        completed_refunds = db.execute(
            select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
            .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
            .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
            .join(Donation, PaymentOrder.donation_id == Donation.id)
            .where(
                and_(
                    Donation.milestone_id == milestone_id,
                    PaymentRefund.status == PaymentRefundStatus.COMPLETED
                )
            )
        ).scalar() or 0

        # 3. Net Raised
        net_raised = max(0, gross_captured - completed_refunds)

        # 4. Committed Payouts (approved or in flight, not yet settled as succeeded)
        committed_payouts = db.execute(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(
                and_(
                    Disbursement.milestone_id == milestone_id,
                    Disbursement.status.in_([
                        DisbursementStatus.APPROVED,
                        DisbursementStatus.SUBMISSION_PENDING,
                        DisbursementStatus.SUBMITTED,
                        DisbursementStatus.PROCESSING
                    ])
                )
            )
        ).scalar() or 0

        # 5. Succeeded Payouts (already disbursed)
        succeeded_payouts = db.execute(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(
                and_(
                    Disbursement.milestone_id == milestone_id,
                    Disbursement.status == DisbursementStatus.SUCCEEDED
                )
            )
        ).scalar() or 0

        # 6. Pending Review Payouts (requested, awaiting approval)
        requested_payouts = db.execute(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(
                and_(
                    Disbursement.milestone_id == milestone_id,
                    Disbursement.status.in_([
                        DisbursementStatus.REQUESTED,
                        DisbursementStatus.PENDING_APPROVAL
                    ])
                )
            )
        ).scalar() or 0

        # 7. Available Balance for new disbursement
        available_balance = max(0, net_raised - committed_payouts - succeeded_payouts)

        target_amount = milestone.target_amount_minor_units or 0
        funding_percentage = (net_raised / target_amount * 100.0) if target_amount > 0 else 0.0

        return {
            "milestone_id": milestone.id,
            "project_id": milestone.project_id,
            "currency": currency,
            "target_amount_minor_units": target_amount,
            "gross_captured_minor_units": gross_captured,
            "completed_refunds_minor_units": completed_refunds,
            "net_raised_minor_units": net_raised,
            "committed_payouts_minor_units": committed_payouts,
            "succeeded_payouts_minor_units": succeeded_payouts,
            "requested_payouts_minor_units": requested_payouts,
            "available_balance_minor_units": available_balance,
            "funding_percentage": round(funding_percentage, 2)
        }

    @staticmethod
    def get_project_fund_summary(
        db: Session,
        project_id: str
    ) -> Dict[str, Any]:
        """Calculates project-wide aggregated financial summary across all milestones."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project:
            raise NotFoundException(f"Project '{project_id}' not found.")

        milestones = db.execute(select(Milestone).where(Milestone.project_id == project_id)).scalars().all()

        total_target = project.target_amount_minor_units or 0
        total_gross = 0
        total_refunds = 0
        total_net = 0
        total_committed = 0
        total_succeeded = 0
        total_available = 0

        milestone_summaries = []
        for ms in milestones:
            ms_sum = FundAllocationService.get_milestone_fund_summary(db, ms.id)
            milestone_summaries.append(ms_sum)
            total_gross += ms_sum["gross_captured_minor_units"]
            total_refunds += ms_sum["completed_refunds_minor_units"]
            total_net += ms_sum["net_raised_minor_units"]
            total_committed += ms_sum["committed_payouts_minor_units"]
            total_succeeded += ms_sum["succeeded_payouts_minor_units"]
            total_available += ms_sum["available_balance_minor_units"]

        funding_percentage = (total_net / total_target * 100.0) if total_target > 0 else 0.0

        return {
            "project_id": project.id,
            "currency": project.currency,
            "total_target_amount_minor_units": total_target,
            "total_gross_captured_minor_units": total_gross,
            "total_completed_refunds_minor_units": total_refunds,
            "total_net_raised_minor_units": total_net,
            "total_committed_payouts_minor_units": total_committed,
            "total_succeeded_payouts_minor_units": total_succeeded,
            "total_available_balance_minor_units": total_available,
            "funding_percentage": round(funding_percentage, 2),
            "milestones": milestone_summaries
        }

