"""
ngo_analytics_service.py - Tenant-Isolated NGO Analytics & Impact Reporting
===========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides organization-level funding, project completion, milestone performance,
and verified evidence analytics strictly isolated to authorized organizations.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, distinct

from backend.app.db.models.users import Organization
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund, Disbursement
)
from backend.app.db.models.evidence import EvidenceFile, VerificationReport
from backend.app.db.models.reviews_impact import ReviewCase, ImpactCertificate
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, DonationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus,
    ReviewCaseStatus
)


class NGOAnalyticsService:
    """Delivers detailed financial, project, and verified impact analytics for NGOs."""

    @classmethod
    def get_ngo_overview(
        cls,
        db: Session,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Returns organization profile, active projects, milestone status, and funding totals."""
        org = db.get(Organization, organization_id)
        if not org:
            raise ValueError(f"Organization '{organization_id}' not found")

        # Base project query
        project_filter = [Project.organization_id == organization_id]
        if start_date:
            project_filter.append(Project.created_at >= start_date)
        if end_date:
            project_filter.append(Project.created_at <= end_date)

        total_projects = db.scalar(select(func.count(Project.id)).where(and_(*project_filter))) or 0
        active_projects = db.scalar(
            select(func.count(Project.id)).where(and_(*project_filter, Project.status == ProjectStatus.ACTIVE))
        ) or 0
        completed_projects = db.scalar(
            select(func.count(Project.id)).where(and_(*project_filter, Project.status == ProjectStatus.COMPLETED))
        ) or 0
        pending_projects = max(0, total_projects - (active_projects + completed_projects))

        # Milestone query joined to organization projects
        total_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(Project.organization_id == organization_id)
        ) or 0
        completed_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(and_(Project.organization_id == organization_id, Milestone.status == MilestoneStatus.COMPLETED))
        ) or 0
        verified_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(and_(Project.organization_id == organization_id, Milestone.status.in_([MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED])))
        ) or 0
        pending_milestones = max(0, total_milestones - completed_milestones)

        # Financial totals
        # Join donations -> project -> organization
        donation_query = (
            select(
                func.coalesce(func.sum(PaymentCapture.amount_minor_units), 0).label("captured"),
                func.count(distinct(Donation.donor_id)).label("unique_donors")
            )
            .select_from(Donation)
            .join(Project, Donation.project_id == Project.id)
            .join(PaymentCapture, PaymentCapture.order_id.in_(
                select(Donation.id)  # captured via payment orders
            ), isouter=True)
            .where(Project.organization_id == organization_id)
        )
        # Direct sum of completed donations for this org's projects
        captured_cents = db.scalar(
            select(func.coalesce(func.sum(Donation.amount_minor_units), 0))
            .join(Project, Donation.project_id == Project.id)
            .where(and_(
                Project.organization_id == organization_id,
                Donation.status == DonationStatus.PAID
            ))
        ) or 0

        # Refunded donations for this org's projects
        refunded_cents = db.scalar(
            select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
            .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
            .join(Donation, Donation.id == PaymentCapture.order_id, isouter=True)
            .join(Project, Donation.project_id == Project.id, isouter=True)
            .where(and_(
                Project.organization_id == organization_id,
                PaymentRefund.status == PaymentRefundStatus.COMPLETED
            ))
        ) or 0
        net_funding_cents = max(0, captured_cents - refunded_cents)

        # Confirmed Disbursements for this org
        disbursed_cents = db.scalar(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(and_(
                Disbursement.organization_id == organization_id,
                Disbursement.status == DisbursementStatus.SUCCEEDED
            ))
        ) or 0

        available_funds_cents = max(0, net_funding_cents - disbursed_cents)

        return {
            "organization_id": org.id,
            "organization_name": org.name,
            "verification_status": org.status.value if hasattr(org.status, "value") else str(org.status),
            "projects": {
                "total": total_projects,
                "active": active_projects,
                "completed": completed_projects,
                "pending": pending_projects
            },
            "milestones": {
                "total": total_milestones,
                "completed": completed_milestones,
                "verified": verified_milestones,
                "pending": pending_milestones
            },
            "financial": {
                "currency": "USD",
                "total_captured_cents": captured_cents,
                "refunded_cents": refunded_cents,
                "net_funding_cents": net_funding_cents,
                "confirmed_disbursements_cents": disbursed_cents,
                "available_funds_cents": available_funds_cents,
                "total_captured_formatted": f"${captured_cents / 100:,.2f}",
                "net_funding_formatted": f"${net_funding_cents / 100:,.2f}",
                "disbursed_formatted": f"${disbursed_cents / 100:,.2f}",
                "available_funds_formatted": f"${available_funds_cents / 100:,.2f}"
            },
            "as_of_date": datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_funding_analytics(
        cls,
        db: Session,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Detailed funding breakdown by project, milestone, time, and donor distribution."""
        # 1. Funding by Project
        project_rows = db.execute(
            select(
                Project.id,
                Project.title,
                Project.target_amount_minor_units,
                func.coalesce(func.sum(Donation.amount_minor_units), 0).label("raised_cents"),
                func.count(Donation.id).label("donation_count")
            )
            .join(Donation, and_(Donation.project_id == Project.id, Donation.status == DonationStatus.PAID), isouter=True)
            .where(Project.organization_id == organization_id)
            .group_by(Project.id, Project.title, Project.target_amount_minor_units)
        ).all()

        funding_by_project = [
            {
                "project_id": row.id,
                "title": row.title,
                "target_amount_cents": row.target_amount_minor_units,
                "raised_amount_cents": row.raised_cents,
                "donation_count": row.donation_count,
                "progress_percent": round((row.raised_cents / row.target_amount_minor_units * 100.0) if row.target_amount_minor_units > 0 else 0.0, 1)
            }
            for row in project_rows
        ]

        # 2. Donor Repeat Statistics
        donor_counts = db.execute(
            select(Donation.donor_id, func.count(Donation.id).label("donation_count"))
            .join(Project, Donation.project_id == Project.id)
            .where(and_(
                Project.organization_id == organization_id,
                Donation.status == DonationStatus.PAID,
                Donation.donor_id != None
            ))
            .group_by(Donation.donor_id)
        ).all()

        total_unique_donors = len(donor_counts)
        repeat_donors = len([d for d in donor_counts if d.donation_count > 1])
        retention_rate = (repeat_donors / total_unique_donors * 100.0) if total_unique_donors > 0 else 0.0

        # 3. Donation Size Distribution
        donations = db.scalars(
            select(Donation.amount_minor_units)
            .join(Project, Donation.project_id == Project.id)
            .where(and_(Project.organization_id == organization_id, Donation.status == DonationStatus.PAID))
        ).all()

        distribution = {
            "under_25": len([a for a in donations if a < 2500]),
            "25_to_100": len([a for a in donations if 2500 <= a < 10000]),
            "100_to_500": len([a for a in donations if 10000 <= a < 50000]),
            "over_500": len([a for a in donations if a >= 50000])
        }

        total_cents = sum(donations)
        avg_donation_cents = int(total_cents / len(donations)) if donations else 0

        return {
            "organization_id": organization_id,
            "total_raised_cents": total_cents,
            "total_donations": len(donations),
            "average_donation_cents": avg_donation_cents,
            "average_donation_formatted": f"${avg_donation_cents / 100:,.2f}",
            "donor_retention": {
                "unique_donors": total_unique_donors,
                "repeat_donors": repeat_donors,
                "retention_rate": round(retention_rate, 1)
            },
            "funding_by_project": funding_by_project,
            "size_distribution": distribution
        }

    @classmethod
    def get_project_performance(cls, db: Session, organization_id: str) -> List[Dict[str, Any]]:
        """Project-level drill-down for all organization projects."""
        projects = db.scalars(
            select(Project).where(Project.organization_id == organization_id).order_by(Project.created_at.desc())
        ).all()

        results = []
        for p in projects:
            milestones = p.milestones or []
            total_m = len(milestones)
            completed_m = len([m for m in milestones if m.status == MilestoneStatus.COMPLETED])
            verified_m = len([m for m in milestones if m.status in [MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED]])

            results.append({
                "project_id": p.id,
                "title": p.title,
                "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                "target_amount_cents": p.target_amount_minor_units,
                "raised_amount_cents": p.raised_amount_minor_units,
                "funding_progress_percent": round((p.raised_amount_minor_units / p.target_amount_minor_units * 100.0) if p.target_amount_minor_units > 0 else 0.0, 1),
                "milestones_total": total_m,
                "milestones_completed": completed_m,
                "milestones_verified": verified_m,
                "completion_percent": round((completed_m / total_m * 100.0) if total_m > 0 else 0.0, 1),
                "created_at": p.created_at.isoformat() if p.created_at else None
            })
        return results

    @classmethod
    def get_evidence_and_verification_analytics(cls, db: Session, organization_id: str) -> Dict[str, Any]:
        """Evidence submission volumes and human reviewer outcomes."""
        evidence_count = db.scalar(
            select(func.count(EvidenceFile.id))
            .join(Milestone, EvidenceFile.milestone_id == Milestone.id)
            .join(Project, Milestone.project_id == Project.id)
            .where(Project.organization_id == organization_id)
        ) or 0

        # Review cases for this organization
        cases = db.scalars(
            select(ReviewCase)
            .join(Milestone, ReviewCase.milestone_id == Milestone.id)
            .join(Project, Milestone.project_id == Project.id)
            .where(Project.organization_id == organization_id)
        ).all()

        approved = len([c for c in cases if c.status == ReviewCaseStatus.APPROVED])
        rejected = len([c for c in cases if c.status == ReviewCaseStatus.REJECTED])
        revisions = len([c for c in cases if c.status == ReviewCaseStatus.REVISION_REQUESTED])
        pending = len([c for c in cases if c.status in [ReviewCaseStatus.PENDING_ASSIGNMENT, ReviewCaseStatus.UNDER_REVIEW]])

        return {
            "evidence_files_submitted": evidence_count,
            "review_cases_total": len(cases),
            "approved": approved,
            "rejected": rejected,
            "revisions_requested": revisions,
            "under_review": pending,
            "approval_rate": round((approved / len(cases) * 100.0) if cases else 0.0, 1)
        }

    @classmethod
    def generate_impact_report(
        cls,
        db: Session,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generates an organization impact report ready for display or PDF/CSV export."""
        overview = cls.get_ngo_overview(db, organization_id, start_date, end_date)
        funding = cls.get_funding_analytics(db, organization_id, start_date, end_date)
        projects = cls.get_project_performance(db, organization_id)
        evidence = cls.get_evidence_and_verification_analytics(db, organization_id)

        # Count certificates issued for this org
        certificates_count = db.scalar(
            select(func.count(ImpactCertificate.id))
            .join(Milestone, ImpactCertificate.milestone_id == Milestone.id)
            .join(Project, Milestone.project_id == Project.id)
            .where(and_(
                Project.organization_id == organization_id,
                ImpactCertificate.status.in_(["ACTIVE", "ISSUED"])
            ))
        ) or 0

        return {
            "report_id": f"REP-NGO-{organization_id[:8]}-{int(datetime.now(timezone.utc).timestamp())}",
            "organization_name": overview["organization_name"],
            "organization_id": organization_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else "All Time",
                "end": end_date.isoformat() if end_date else "Present"
            },
            "overview": overview,
            "funding_summary": funding,
            "projects_summary": projects,
            "evidence_and_verification": evidence,
            "verified_impact": {
                "certificates_issued": certificates_count,
                "verified_milestones": overview["milestones"]["verified"],
                "completed_milestones": overview["milestones"]["completed"]
            },
            "disclaimer": "This impact report reflects verified records recorded in the CharityChain ledger. Data is subject to ongoing human and automated verification."
        }
