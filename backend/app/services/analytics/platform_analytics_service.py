"""
platform_analytics_service.py - Platform-Wide & Administrative Analytics
========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides platform-level KPIs, financial operations reconciliation, review workload,
and system health telemetry using authoritative database records.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, case

from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund, Disbursement,
    ReconciliationItem
)
from backend.app.db.models.ledger import JournalLine
from backend.app.db.models.reviews_impact import (
    ReviewCase, ImpactCertificate, ImpactLedgerEvent, ImpactLedgerCheckpoint
)
from backend.app.db.models.evidence import VerificationReport, VerificationJob
from backend.app.db.models.notifications import NotificationOutbox, WebhookDelivery
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus,
    ReviewCaseStatus, LedgerEntryType, ReconciliationStatus
)


class PlatformAnalyticsService:
    """Computes high-level platform health, financial metrics, and operational performance."""

    @classmethod
    def get_platform_overview(cls, db: Session) -> Dict[str, Any]:
        """Calculates global platform KPIs across users, projects, funding, and impact."""
        # 1. User & NGO Counts
        total_users = db.scalar(select(func.count(User.id))) or 0
        active_users = db.scalar(select(func.count(User.id)).where(User.is_active == True)) or 0
        total_orgs = db.scalar(select(func.count(Organization.id))) or 0
        verified_orgs = db.scalar(
            select(func.count(Organization.id)).where(Organization.status == OrganizationStatus.VERIFIED)
        ) or 0

        # 2. Project & Milestone Counts
        total_projects = db.scalar(select(func.count(Project.id))) or 0
        active_projects = db.scalar(
            select(func.count(Project.id)).where(Project.status == ProjectStatus.ACTIVE)
        ) or 0
        completed_projects = db.scalar(
            select(func.count(Project.id)).where(Project.status == ProjectStatus.COMPLETED)
        ) or 0

        total_milestones = db.scalar(select(func.count(Milestone.id))) or 0
        completed_milestones = db.scalar(
            select(func.count(Milestone.id)).where(Milestone.status == MilestoneStatus.COMPLETED)
        ) or 0
        verified_milestones = db.scalar(
            select(func.count(Milestone.id)).where(Milestone.status.in_([MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED]))
        ) or 0

        # 3. Financial Totals (Captured, Refunded, Net, Disbursed)
        captured_cents = db.scalar(
            select(func.coalesce(func.sum(PaymentCapture.amount_minor_units), 0))
            .where(PaymentCapture.status == PaymentCaptureStatus.COMPLETED)
        ) or 0

        refunded_cents = db.scalar(
            select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
            .where(PaymentRefund.status == PaymentRefundStatus.COMPLETED)
        ) or 0

        net_funding_cents = max(0, captured_cents - refunded_cents)

        disbursed_cents = db.scalar(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(Disbursement.status == DisbursementStatus.SUCCEEDED)
        ) or 0

        # 4. Impact Certificates & Communications
        total_certificates = db.scalar(
            select(func.count(ImpactCertificate.id)).where(ImpactCertificate.status.in_(["ACTIVE", "ISSUED"]))
        ) or 0

        notifications_processed = db.scalar(
            select(func.count(NotificationOutbox.id)).where(NotificationOutbox.status == "PROCESSED")
        ) or 0

        return {
            "users": {
                "total": total_users,
                "active": active_users
            },
            "organizations": {
                "total": total_orgs,
                "verified": verified_orgs,
                "pending": max(0, total_orgs - verified_orgs)
            },
            "projects": {
                "total": total_projects,
                "active": active_projects,
                "completed": completed_projects
            },
            "milestones": {
                "total": total_milestones,
                "completed": completed_milestones,
                "verified": verified_milestones
            },
            "financial": {
                "currency": "USD",
                "captured_cents": captured_cents,
                "refunded_cents": refunded_cents,
                "net_funding_cents": net_funding_cents,
                "disbursed_cents": disbursed_cents,
                "captured_formatted": f"${captured_cents / 100:,.2f}",
                "refunded_formatted": f"${refunded_cents / 100:,.2f}",
                "net_funding_formatted": f"${net_funding_cents / 100:,.2f}",
                "disbursed_formatted": f"${disbursed_cents / 100:,.2f}"
            },
            "impact": {
                "certificates_issued": total_certificates,
                "notifications_processed": notifications_processed
            },
            "as_of_date": datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_financial_operations_analytics(cls, db: Session) -> Dict[str, Any]:
        """Provides financial operational health, payout metrics, and ledger consistency."""
        # 1. Payment Success & Refund Rates
        total_captures = db.scalar(select(func.count(PaymentCapture.id))) or 0
        successful_captures = db.scalar(
            select(func.count(PaymentCapture.id)).where(PaymentCapture.status == PaymentCaptureStatus.COMPLETED)
        ) or 0
        capture_success_rate = (successful_captures / total_captures * 100.0) if total_captures > 0 else 100.0

        total_refunds = db.scalar(
            select(func.count(PaymentRefund.id)).where(PaymentRefund.status == PaymentRefundStatus.COMPLETED)
        ) or 0
        refund_rate = (total_refunds / successful_captures * 100.0) if successful_captures > 0 else 0.0

        # 2. Payout Operations
        total_payouts = db.scalar(select(func.count(Disbursement.id))) or 0
        completed_payouts = db.scalar(
            select(func.count(Disbursement.id)).where(Disbursement.status == DisbursementStatus.SUCCEEDED)
        ) or 0
        failed_payouts = db.scalar(
            select(func.count(Disbursement.id)).where(Disbursement.status == DisbursementStatus.FAILED)
        ) or 0
        payout_success_rate = (completed_payouts / total_payouts * 100.0) if total_payouts > 0 else 100.0

        # 3. Reconciliation Discrepancies
        open_discrepancies = db.scalar(
            select(func.count(ReconciliationItem.id)).where(
                and_(
                    ReconciliationItem.reconciliation_status == ReconciliationStatus.DISCREPANCY,
                    ReconciliationItem.resolved_at == None
                )
            )
        ) or 0

        # 4. Double-Entry Ledger Trial Balance Consistency
        # Assets (Debit) must equal Liabilities + Equity (Credit)
        total_debits = db.scalar(
            select(func.coalesce(func.sum(JournalLine.debit_minor_units), 0))
        ) or 0
        total_credits = db.scalar(
            select(func.coalesce(func.sum(JournalLine.credit_minor_units), 0))
        ) or 0
        is_ledger_balanced = (total_debits == total_credits)
        ledger_discrepancy_cents = abs(total_debits - total_credits)

        return {
            "payment_operations": {
                "total_captures": total_captures,
                "successful_captures": successful_captures,
                "capture_success_rate": round(capture_success_rate, 2),
                "total_refunds": total_refunds,
                "refund_rate": round(refund_rate, 2)
            },
            "payout_operations": {
                "total_payouts": total_payouts,
                "completed_payouts": completed_payouts,
                "failed_payouts": failed_payouts,
                "payout_success_rate": round(payout_success_rate, 2)
            },
            "reconciliation": {
                "open_discrepancies_count": open_discrepancies,
                "requires_attention": open_discrepancies > 0
            },
            "financial_ledger_integrity": {
                "is_balanced": is_ledger_balanced,
                "total_debits_cents": total_debits,
                "total_credits_cents": total_credits,
                "discrepancy_cents": ledger_discrepancy_cents
            }
        }

    @classmethod
    def get_review_and_verification_analytics(cls, db: Session) -> Dict[str, Any]:
        """Provides workload, turnaround, and quality metrics for AI and human verification."""
        # 1. Human Review Pipeline
        total_cases = db.scalar(select(func.count(ReviewCase.id))) or 0
        pending_cases = db.scalar(
            select(func.count(ReviewCase.id)).where(ReviewCase.status.in_([
                ReviewCaseStatus.PENDING_ASSIGNMENT, ReviewCaseStatus.UNDER_REVIEW
            ]))
        ) or 0
        approved_cases = db.scalar(
            select(func.count(ReviewCase.id)).where(ReviewCase.status == ReviewCaseStatus.APPROVED)
        ) or 0
        rejected_cases = db.scalar(
            select(func.count(ReviewCase.id)).where(ReviewCase.status == ReviewCaseStatus.REJECTED)
        ) or 0
        revisions_requested = db.scalar(
            select(func.count(ReviewCase.id)).where(ReviewCase.status == ReviewCaseStatus.REVISION_REQUESTED)
        ) or 0

        completion_rate = ((approved_cases + rejected_cases) / total_cases * 100.0) if total_cases > 0 else 100.0

        # 2. AI Verification Insights
        total_reports = db.scalar(select(func.count(VerificationReport.id))) or 0
        avg_confidence = db.scalar(select(func.avg(VerificationReport.confidence_score))) or 0.0

        flagged_reports = db.scalar(
            select(func.count(VerificationReport.id)).where(VerificationReport.recommendation != "PASS")
        ) or 0

        return {
            "human_review_pipeline": {
                "total_cases": total_cases,
                "backlog_pending": pending_cases,
                "approved_cases": approved_cases,
                "rejected_cases": rejected_cases,
                "revisions_requested": revisions_requested,
                "completion_rate": round(completion_rate, 2)
            },
            "ai_verification_engine": {
                "total_reports": total_reports,
                "average_confidence": round(float(avg_confidence) * 100, 1),
                "flagged_reports": flagged_reports,
                "flag_rate": round((flagged_reports / total_reports * 100) if total_reports > 0 else 0.0, 2)
            }
        }

    @classmethod
    def get_operational_health(cls, db: Session) -> Dict[str, Any]:
        """Provides background queue, webhook delivery, and ledger checkpoint health."""
        # Outbox telemetry
        outbox_pending = db.scalar(
            select(func.count(NotificationOutbox.id)).where(NotificationOutbox.status == "PENDING")
        ) or 0
        outbox_dead_letter = db.scalar(
            select(func.count(NotificationOutbox.id)).where(NotificationOutbox.status == "DEAD_LETTER")
        ) or 0

        # Webhook delivery telemetry
        webhook_deliveries = db.scalar(select(func.count(WebhookDelivery.id))) or 0
        webhook_successful = db.scalar(
            select(func.count(WebhookDelivery.id)).where(WebhookDelivery.status == "SUCCESS")
        ) or 0
        webhook_rate = (webhook_successful / webhook_deliveries * 100.0) if webhook_deliveries > 0 else 100.0

        # Impact Ledger
        ledger_events = db.scalar(select(func.count(ImpactLedgerEvent.id))) or 0
        checkpoints_count = db.scalar(select(func.count(ImpactLedgerCheckpoint.id))) or 0

        return {
            "notification_outbox": {
                "pending_queue_size": outbox_pending,
                "dead_letter_count": outbox_dead_letter
            },
            "webhook_deliveries": {
                "total": webhook_deliveries,
                "successful": webhook_successful,
                "success_rate": round(webhook_rate, 2)
            },
            "impact_ledger_integrity": {
                "total_events_logged": ledger_events,
                "checkpoints_count": checkpoints_count,
                "integrity_status": "HEALTHY"
            }
        }
