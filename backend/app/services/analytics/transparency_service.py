"""
transparency_service.py - Public Transparency & Verifiable Impact Portal
========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides unauthenticated, privacy-safe, authoritative visibility into platform metrics,
project progress, NGO profiles, and cryptographic proof-of-impact certificates.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, distinct

from backend.app.db.models.users import Organization
from backend.app.db.models.projects import Project, Milestone, ProjectCategory
from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund, Disbursement
)
from backend.app.db.models.evidence import EvidenceFile
from backend.app.db.models.reviews_impact import (
    ImpactCertificate, ImpactLedgerEvent
)
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus,
    ProjectVisibility, PaymentCaptureStatus, PaymentRefundStatus,
    DisbursementStatus
)


class PublicTransparencyService:
    """Delivers privacy-conscious, authoritative public transparency analytics."""

    @classmethod
    def get_global_overview(cls, db: Session) -> Dict[str, Any]:
        """Global platform summary safe for public and unauthenticated inspection."""
        # Organizations
        total_orgs = db.scalar(select(func.count(Organization.id))) or 0
        verified_orgs = db.scalar(
            select(func.count(Organization.id)).where(Organization.status == OrganizationStatus.VERIFIED)
        ) or 0

        # Public Projects
        public_filter = [Project.public_visibility == ProjectVisibility.PUBLIC]
        total_published_projects = db.scalar(select(func.count(Project.id)).where(and_(*public_filter))) or 0
        active_projects = db.scalar(
            select(func.count(Project.id)).where(and_(*public_filter, Project.status == ProjectStatus.ACTIVE))
        ) or 0
        completed_projects = db.scalar(
            select(func.count(Project.id)).where(and_(*public_filter, Project.status == ProjectStatus.COMPLETED))
        ) or 0

        # Milestones
        total_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(Project.public_visibility == ProjectVisibility.PUBLIC)
        ) or 0
        completed_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(and_(
                Project.public_visibility == ProjectVisibility.PUBLIC,
                Milestone.status == MilestoneStatus.COMPLETED
            ))
        ) or 0
        verified_milestones = db.scalar(
            select(func.count(Milestone.id))
            .join(Project, Milestone.project_id == Project.id)
            .where(and_(
                Project.public_visibility == ProjectVisibility.PUBLIC,
                Milestone.status.in_([MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED])
            ))
        ) or 0

        # Financial Totals
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

        # Total active certificates
        certificates_count = db.scalar(
            select(func.count(ImpactCertificate.id))
            .where(ImpactCertificate.status.in_(["ACTIVE", "ISSUED"]))
        ) or 0

        return {
            "organizations": {
                "total_listed": total_orgs,
                "verified": verified_orgs
            },
            "projects": {
                "published": total_published_projects,
                "active": active_projects,
                "completed": completed_projects
            },
            "milestones": {
                "total": total_milestones,
                "completed": completed_milestones,
                "verified": verified_milestones
            },
            "financial_transparency": {
                "currency": "USD",
                "captured_cents": captured_cents,
                "refunded_cents": refunded_cents,
                "net_donations_cents": net_funding_cents,
                "confirmed_disbursements_cents": disbursed_cents,
                "captured_formatted": f"${captured_cents / 100:,.2f}",
                "net_donations_formatted": f"${net_funding_cents / 100:,.2f}",
                "disbursed_formatted": f"${disbursed_cents / 100:,.2f}"
            },
            "impact_certificates_issued": certificates_count,
            "as_of_date": datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_public_projects(
        cls,
        db: Session,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lists publicly visible projects with progress metrics and milestone completion."""
        query = select(Project).where(Project.public_visibility == ProjectVisibility.PUBLIC)

        if category:
            query = query.join(ProjectCategory, Project.category_id == ProjectCategory.id).where(
                ProjectCategory.name.ilike(f"%{category}%")
            )
        if status:
            query = query.where(Project.status == status)

        query = query.order_by(Project.created_at.desc()).limit(limit).offset(offset)
        projects = db.scalars(query).all()

        results = []
        for p in projects:
            milestones = p.milestones or []
            total_m = len(milestones)
            completed_m = len([m for m in milestones if m.status == MilestoneStatus.COMPLETED])
            verified_m = len([m for m in milestones if m.status in [MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED]])

            results.append({
                "project_id": p.id,
                "title": p.title,
                "summary": p.summary,
                "organization_name": p.organization.name if p.organization else "Verified NGO",
                "category": p.category.name if p.category else "General",
                "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                "target_amount_cents": p.target_amount_minor_units,
                "raised_amount_cents": p.raised_amount_minor_units,
                "target_formatted": f"${p.target_amount_minor_units / 100:,.2f}",
                "raised_formatted": f"${p.raised_amount_minor_units / 100:,.2f}",
                "funding_percent": round((p.raised_amount_minor_units / p.target_amount_minor_units * 100.0) if p.target_amount_minor_units > 0 else 0.0, 1),
                "milestones_total": total_m,
                "milestones_completed": completed_m,
                "milestones_verified": verified_m,
                "country": p.country,
                "city": p.city
            })
        return results

    @classmethod
    def get_public_project_details(cls, db: Session, project_id: str) -> Optional[Dict[str, Any]]:
        """Deep public transparency page for a specific project."""
        p = db.get(Project, project_id)
        if not p or p.public_visibility != ProjectVisibility.PUBLIC:
            return None

        milestones_data = []
        for m in p.milestones or []:
            evidence_count = len(m.evidence_files) if m.evidence_files else 0
            milestones_data.append({
                "milestone_id": m.id,
                "sequence_number": m.sequence_number,
                "title": m.title,
                "description": m.description,
                "target_amount_cents": m.target_amount_minor_units,
                "raised_amount_cents": m.raised_amount_minor_units,
                "target_formatted": f"${m.target_amount_minor_units / 100:,.2f}",
                "status": m.status.value if hasattr(m.status, "value") else str(m.status),
                "evidence_files_count": evidence_count,
                "is_verified": m.status in [MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED]
            })

        # Fetch publicly verifiable certificates for this project
        certs = db.scalars(
            select(ImpactCertificate)
            .join(Milestone, ImpactCertificate.milestone_id == Milestone.id)
            .where(and_(
                Milestone.project_id == project_id,
                ImpactCertificate.status.in_(["ACTIVE", "ISSUED"])
            ))
        ).all()

        certificates_data = [
            {
                "certificate_code": c.certificate_code,
                "verification_token": c.verification_token,
                "milestone_id": c.milestone_id,
                "issued_at": c.issued_at.isoformat() if c.issued_at else None
            }
            for c in certs[:10]  # Public preview of recent certificates
        ]

        return {
            "project_id": p.id,
            "title": p.title,
            "summary": p.summary,
            "description": p.description,
            "objectives": p.objectives,
            "organization": {
                "id": p.organization_id,
                "name": p.organization.name if p.organization else "Verified NGO",
                "verification_status": p.organization.status.value if (p.organization and hasattr(p.organization.status, "value")) else "VERIFIED"
            },
            "category": p.category.name if p.category else "Charitable",
            "status": p.status.value if hasattr(p.status, "value") else str(p.status),
            "target_amount_cents": p.target_amount_minor_units,
            "raised_amount_cents": p.raised_amount_minor_units,
            "target_formatted": f"${p.target_amount_minor_units / 100:,.2f}",
            "raised_formatted": f"${p.raised_amount_minor_units / 100:,.2f}",
            "currency": p.currency,
            "milestones": milestones_data,
            "certificates": certificates_data,
            "country": p.country,
            "city": p.city,
            "updated_at": p.updated_at.isoformat() if p.updated_at else datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_public_organization_profile(cls, db: Session, organization_id: str) -> Optional[Dict[str, Any]]:
        """Returns public-safe profile and aggregate stats for an NGO."""
        org = db.get(Organization, organization_id)
        if not org:
            return None

        projects = db.scalars(
            select(Project).where(and_(
                Project.organization_id == organization_id,
                Project.public_visibility == ProjectVisibility.PUBLIC
            ))
        ).all()

        total_p = len(projects)
        completed_p = len([p for p in projects if p.status == ProjectStatus.COMPLETED])
        active_p = len([p for p in projects if p.status == ProjectStatus.ACTIVE])

        total_raised_cents = sum(p.raised_amount_minor_units for p in projects)

        # Confirmed completed payouts to this NGO
        disbursed_cents = db.scalar(
            select(func.coalesce(func.sum(Disbursement.amount_minor_units), 0))
            .where(and_(
                Disbursement.organization_id == organization_id,
                Disbursement.status == DisbursementStatus.SUCCEEDED
            ))
        ) or 0

        return {
            "organization_id": org.id,
            "name": org.name,
            "status": org.status.value if hasattr(org.status, "value") else str(org.status),
            "country": org.country if hasattr(org, "country") else "USA",
            "total_public_projects": total_p,
            "active_projects": active_p,
            "completed_projects": completed_p,
            "total_raised_cents": total_raised_cents,
            "total_raised_formatted": f"${total_raised_cents / 100:,.2f}",
            "confirmed_disbursements_cents": disbursed_cents,
            "confirmed_disbursements_formatted": f"${disbursed_cents / 100:,.2f}"
        }

    @classmethod
    def get_methodology(cls) -> Dict[str, Any]:
        """Explains public metric calculation methodology and privacy exclusions."""
        return {
            "methodology_version": "v1.0.0",
            "principles": [
                "AI analyzes. Evidence supports. Humans decide. Payment systems execute. Impact records remain auditable. Analytics explain. Transparency builds trust."
            ],
            "financial_calculation": {
                "captured_donations": "Represents settled funds captured through PayPal REST gateway. Authorized-only holds are excluded.",
                "refunds": "All completed donor refunds are deducted from net donation totals.",
                "disbursements": "Only confirmed payouts sent to verified NGO recipient accounts are counted toward completed disbursements.",
                "currency_convention": "All transactions are stored and reconciled in integer minor units (cents) to eliminate binary floating-point rounding errors."
            },
            "verification_and_review": {
                "ai_evidence_verification": "AI forensic analysis (ELA, pHash, YOLOv8) produces advisory confidence signals. AI confidence is not treated as verified ground truth.",
                "human_review": "Milestone fund release requires dual-key independent human reviewer approval.",
                "certificates": "Cryptographically signed Proof-of-Impact certificates (Ed25519/RSA-PSS) are issued upon milestone completion and independently verifiable."
            },
            "privacy_and_exclusions": {
                "excluded_data": "Individual donor names, email addresses, credit card tokens, exact shelter/medical beneficiary coordinates, and internal reviewer communications are strictly excluded from public metrics."
            },
            "refresh_cadence": "Platform statistics are computed directly against authoritative database records and reflect current system state."
        }
