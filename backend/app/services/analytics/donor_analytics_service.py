"""
donor_analytics_service.py - Personalized Donor Portfolio & Impact Tracking
===========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides personalized giving portfolios, multi-currency contribution tracking,
and full end-to-end impact journeys for individual donors.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, distinct

from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund
)
from backend.app.db.models.projects import Project, Milestone, ProjectCategory
from backend.app.db.models.users import User, Organization
from backend.app.db.models.evidence import EvidenceFile, VerificationReport
from backend.app.db.models.reviews_impact import ReviewCase, ImpactCertificate
from backend.app.core.constants import (
    DonationStatus, PaymentCaptureStatus, PaymentRefundStatus,
    MilestoneStatus, ProjectStatus
)


class DonorAnalyticsService:
    """Manages individual donor portfolios and verifiable impact attribution."""

    @classmethod
    def get_donor_overview(cls, db: Session, donor_id: str) -> Dict[str, Any]:
        """Provides high-level portfolio metrics for a specific donor."""
        donations = db.scalars(
            select(Donation).where(
                and_(Donation.donor_id == donor_id, Donation.status == DonationStatus.PAID)
            )
        ).all()

        if not donations:
            return {
                "donor_id": donor_id,
                "total_captured_cents": 0,
                "refunded_cents": 0,
                "net_contributions_cents": 0,
                "total_captured_formatted": "$0.00",
                "net_contributions_formatted": "$0.00",
                "supported_projects_count": 0,
                "supported_ngos_count": 0,
                "completed_milestones_count": 0,
                "certificates_count": 0,
                "currencies": {},
                "as_of_date": datetime.now(timezone.utc).isoformat()
            }

        # Multi-currency aggregation
        currencies: Dict[str, Dict[str, int]] = {}
        project_ids = set()
        milestone_ids = set()

        for d in donations:
            cur = d.currency or "USD"
            if cur not in currencies:
                currencies[cur] = {"captured_cents": 0, "refunded_cents": 0, "net_cents": 0}
            currencies[cur]["captured_cents"] += d.amount_minor_units
            project_ids.add(d.project_id)
            milestone_ids.add(d.milestone_id)

        # Count supported NGOs
        supported_orgs = db.scalar(
            select(func.count(distinct(Project.organization_id)))
            .where(Project.id.in_(project_ids))
        ) or 0

        # Count completed milestones supported
        completed_m = db.scalar(
            select(func.count(distinct(Milestone.id)))
            .where(and_(
                Milestone.id.in_(milestone_ids),
                Milestone.status == MilestoneStatus.COMPLETED
            ))
        ) or 0

        # Count active certificates earned
        certs_count = db.scalar(
            select(func.count(ImpactCertificate.id))
            .join(Donation, ImpactCertificate.donation_id == Donation.id)
            .where(and_(
                Donation.donor_id == donor_id,
                ImpactCertificate.status.in_(["ACTIVE", "ISSUED"])
            ))
        ) or 0

        # Compute USD totals (or primary currency)
        usd_data = currencies.get("USD", {"captured_cents": 0, "refunded_cents": 0, "net_cents": 0})
        usd_data["net_cents"] = usd_data["captured_cents"] - usd_data["refunded_cents"]

        return {
            "donor_id": donor_id,
            "total_captured_cents": usd_data["captured_cents"],
            "refunded_cents": usd_data["refunded_cents"],
            "net_contributions_cents": usd_data["net_cents"],
            "total_captured_formatted": f"${usd_data['captured_cents'] / 100:,.2f}",
            "net_contributions_formatted": f"${usd_data['net_cents'] / 100:,.2f}",
            "supported_projects_count": len(project_ids),
            "supported_ngos_count": supported_orgs,
            "completed_milestones_count": completed_m,
            "certificates_count": certs_count,
            "currencies": currencies,
            "as_of_date": datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_donation_portfolio(cls, db: Session, donor_id: str) -> Dict[str, Any]:
        """Detailed breakdown by category, organization, and donation size."""
        donations = db.scalars(
            select(Donation)
            .where(and_(Donation.donor_id == donor_id, Donation.status == DonationStatus.PAID))
            .order_by(Donation.created_at.desc())
        ).all()

        # Category breakdown
        category_map: Dict[str, int] = {}
        org_map: Dict[str, Dict[str, Any]] = {}
        donation_history = []

        for d in donations:
            amt = d.amount_minor_units
            proj = d.project
            cat_name = proj.category.name if (proj and proj.category) else "General Charitable"
            org_name = proj.organization.name if (proj and proj.organization) else "Verified NGO"
            org_id = proj.organization_id if proj else "unknown"

            category_map[cat_name] = category_map.get(cat_name, 0) + amt

            if org_id not in org_map:
                org_map[org_id] = {"name": org_name, "total_cents": 0, "count": 0}
            org_map[org_id]["total_cents"] += amt
            org_map[org_id]["count"] += 1

            donation_history.append({
                "donation_id": d.id,
                "project_id": d.project_id,
                "project_title": proj.title if proj else "Charity Project",
                "milestone_id": d.milestone_id,
                "amount_cents": d.amount_minor_units,
                "amount_formatted": f"${d.amount_minor_units / 100:,.2f}",
                "currency": d.currency,
                "created_at": d.created_at.isoformat() if d.created_at else None
            })

        total_cents = sum(d.amount_minor_units for d in donations)
        avg_donation_cents = int(total_cents / len(donations)) if donations else 0

        return {
            "donor_id": donor_id,
            "total_donations": len(donations),
            "total_given_cents": total_cents,
            "average_donation_cents": avg_donation_cents,
            "by_category": [
                {"category": cat, "amount_cents": cents, "amount_formatted": f"${cents / 100:,.2f}"}
                for cat, cents in category_map.items()
            ],
            "by_organization": [
                {"organization_id": oid, "name": data["name"], "amount_cents": data["total_cents"], "count": data["count"]}
                for oid, data in org_map.items()
            ],
            "recent_donations": donation_history[:20]
        }

    @classmethod
    def get_personal_impact_journey(cls, db: Session, donor_id: str) -> List[Dict[str, Any]]:
        """
        Maps each donor contribution to its full lifecycle:
        Donation -> Project -> Milestone -> Evidence -> AI Analysis -> Human Review -> Certificate.
        """
        donations = db.scalars(
            select(Donation)
            .where(and_(Donation.donor_id == donor_id, Donation.status == DonationStatus.PAID))
            .order_by(Donation.created_at.desc())
        ).all()

        journeys = []
        for d in donations:
            proj = d.project
            m = d.milestone
            cert = d.certificate

            # Look for evidence on this milestone
            evidence_count = len(m.evidence_files) if (m and m.evidence_files) else 0

            # Status description
            milestone_status = m.status.value if (m and hasattr(m.status, "value")) else str(m.status if m else "PENDING")
            has_certificate = cert is not None and getattr(cert, "status", None) in ["ACTIVE", "ISSUED"]

            journeys.append({
                "donation_id": d.id,
                "amount_cents": d.amount_minor_units,
                "amount_formatted": f"${d.amount_minor_units / 100:,.2f}",
                "currency": d.currency,
                "donated_at": d.created_at.isoformat() if d.created_at else None,
                "project": {
                    "id": proj.id if proj else None,
                    "title": proj.title if proj else "Charity Project",
                    "status": proj.status.value if (proj and hasattr(proj.status, "value")) else "ACTIVE"
                },
                "milestone": {
                    "id": m.id if m else None,
                    "title": m.title if m else "Project Milestone",
                    "status": milestone_status,
                    "sequence_number": m.sequence_number if m else 1
                },
                "evidence_submitted": evidence_count > 0,
                "evidence_count": evidence_count,
                "is_completed": milestone_status in ["RELEASED", "COMPLETED"],
                "certificate": {
                    "has_certificate": has_certificate,
                    "certificate_id": cert.id if cert else None,
                    "certificate_code": getattr(cert, "certificate_number", getattr(cert, "certificate_code", None)) if cert else None,
                    "verification_token": getattr(cert, "verification_id", getattr(cert, "verification_token", None)) if cert else None
                }
            })
        return journeys

    @classmethod
    def generate_annual_giving_report(
        cls, db: Session, donor_id: str, year: int
    ) -> Dict[str, Any]:
        """Generates an annual charitable contribution summary with tax disclosure disclaimers."""
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        donations = db.scalars(
            select(Donation)
            .where(and_(
                Donation.donor_id == donor_id,
                Donation.status == DonationStatus.PAID,
                Donation.created_at >= start_date,
                Donation.created_at <= end_date
            ))
            .order_by(Donation.created_at.asc())
        ).all()

        total_cents = sum(d.amount_minor_units for d in donations)
        donor = db.scalar(select(User).where(User.id == donor_id))

        items = [
            {
                "date": d.created_at.strftime("%Y-%m-%d") if d.created_at else "N/A",
                "donation_id": d.id,
                "project_title": d.project.title if d.project else "Charity Project",
                "organization_name": d.project.organization.name if (d.project and d.project.organization) else "Verified NGO",
                "amount_cents": d.amount_minor_units,
                "amount_formatted": f"${d.amount_minor_units / 100:,.2f}",
                "currency": d.currency
            }
            for d in donations
        ]

        return {
            "report_id": f"TAX-ANNUAL-{donor_id[:8]}-{year}",
            "donor_id": donor_id,
            "tax_year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_contributions_cents": total_cents,
            "total_contributions_formatted": f"${total_cents / 100:,.2f}",
            "currency": "USD",
            "donation_count": len(donations),
            "contributions": items,
            "tax_disclaimer": "This document is a statement of charitable contributions recorded on CharityChain. It does not constitute official tax advice. Please consult with your tax advisor regarding deductibility."
        }
