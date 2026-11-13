"""
analytics.py - Role-Authorized Analytics API Endpoints
======================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Exposes endpoints for NGO analytics, Donor portfolios, Platform administration,
and secure CSV/report exports with strict tenant isolation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, Query, Path, Response, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import StandardResponse
from backend.app.schemas.analytics import (
    NGOOverviewResponse, NGOFundingResponse, ProjectPerformanceItem,
    EvidenceVerificationAnalyticsResponse, DonorOverviewResponse,
    DonorPortfolioResponse, DonorImpactJourneyItem, AnnualGivingReportResponse,
    PlatformOverviewResponse, FinancialOperationsResponse,
    ReviewVerificationOperationsResponse, OperationalHealthResponse
)
from backend.app.core.constants import PlatformRole, OrganizationRole
from backend.app.core.permissions import (
    get_current_user, get_current_active_user, require_platform_role, require_org_role
)
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.services.analytics.ngo_analytics_service import NGOAnalyticsService
from backend.app.services.analytics.donor_analytics_service import DonorAnalyticsService
from backend.app.services.analytics.platform_analytics_service import PlatformAnalyticsService
from backend.app.services.analytics.analytics_export_service import AnalyticsExportService

router = APIRouter(prefix="/analytics", tags=["Analytics & Impact Reporting"])


# ============================================================================
# NGO Analytics Endpoints (Tenant-Isolated)
# ============================================================================

def verify_ngo_access(
    organization_id: str,
    current_user: User,
    db: Session
):
    """Ensures user is an authorized member/admin of the organization or a platform admin."""
    require_org_role(
        org_id=organization_id,
        user_id=current_user.id,
        db=db,
        allowed_roles=[OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER]
    )


@router.get("/ngo/{organization_id}/overview", response_model=StandardResponse[NGOOverviewResponse])
def get_ngo_overview(
    organization_id: str = Path(..., description="Organization UUID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns organization profile, project counts, milestone status, and funding totals."""
    verify_ngo_access(organization_id, current_user, db)
    data = NGOAnalyticsService.get_ngo_overview(db, organization_id, start_date, end_date)
    return StandardResponse(data=NGOOverviewResponse(**data))


@router.get("/ngo/{organization_id}/funding", response_model=StandardResponse[NGOFundingResponse])
def get_ngo_funding(
    organization_id: str = Path(..., description="Organization UUID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns detailed funding analytics by project, milestone, and donor distribution."""
    verify_ngo_access(organization_id, current_user, db)
    data = NGOAnalyticsService.get_funding_analytics(db, organization_id, start_date, end_date)
    return StandardResponse(data=NGOFundingResponse(**data))


@router.get("/ngo/{organization_id}/projects", response_model=StandardResponse[List[ProjectPerformanceItem]])
def get_ngo_projects(
    organization_id: str = Path(..., description="Organization UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns milestone and funding progress drill-downs for all organization projects."""
    verify_ngo_access(organization_id, current_user, db)
    data = NGOAnalyticsService.get_project_performance(db, organization_id)
    return StandardResponse(data=[ProjectPerformanceItem(**p) for p in data])


@router.get("/ngo/{organization_id}/evidence", response_model=StandardResponse[EvidenceVerificationAnalyticsResponse])
def get_ngo_evidence_analytics(
    organization_id: str = Path(..., description="Organization UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns evidence submission volumes and human reviewer outcomes."""
    verify_ngo_access(organization_id, current_user, db)
    data = NGOAnalyticsService.get_evidence_and_verification_analytics(db, organization_id)
    return StandardResponse(data=EvidenceVerificationAnalyticsResponse(**data))


@router.get("/ngo/{organization_id}/impact-report", response_model=StandardResponse[Dict[str, Any]])
def get_ngo_impact_report(
    organization_id: str = Path(..., description="Organization UUID"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generates an organization impact report ready for presentation or export."""
    verify_ngo_access(organization_id, current_user, db)
    report = NGOAnalyticsService.generate_impact_report(db, organization_id, start_date, end_date)
    return StandardResponse(data=report)


# ============================================================================
# Donor Analytics Endpoints (Personal Isolation)
# ============================================================================

@router.get("/donor/overview", response_model=StandardResponse[DonorOverviewResponse])
def get_donor_overview(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns high-level giving overview and multi-currency totals for the current donor."""
    data = DonorAnalyticsService.get_donor_overview(db, current_user.id)
    return StandardResponse(data=DonorOverviewResponse(**data))


@router.get("/donor/portfolio", response_model=StandardResponse[DonorPortfolioResponse])
def get_donor_portfolio(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Returns contribution breakdown by category, supported organization, and recent donations."""
    data = DonorAnalyticsService.get_donation_portfolio(db, current_user.id)
    return StandardResponse(data=DonorPortfolioResponse(**data))


@router.get("/donor/impact", response_model=StandardResponse[List[DonorImpactJourneyItem]])
def get_donor_impact_journey(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Traces each donation through milestone verification and issued impact certificates."""
    data = DonorAnalyticsService.get_personal_impact_journey(db, current_user.id)
    return StandardResponse(data=[DonorImpactJourneyItem(**j) for j in data])


@router.get("/donor/annual-report", response_model=StandardResponse[AnnualGivingReportResponse])
def get_donor_annual_report(
    year: int = Query(2026, ge=2020, le=2050, description="Tax year"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generates an annual giving report with required tax disclosure disclaimers."""
    data = DonorAnalyticsService.generate_annual_giving_report(db, current_user.id, year)
    return StandardResponse(data=AnnualGivingReportResponse(**data))


# ============================================================================
# Platform & Administrative Analytics Endpoints
# ============================================================================

@router.get(
    "/admin/overview",
    response_model=StandardResponse[PlatformOverviewResponse],
    dependencies=[Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))]
)
def get_admin_platform_overview(db: Session = Depends(get_db)):
    """Provides high-level platform health, total users, projects, funding, and certificates."""
    data = PlatformAnalyticsService.get_platform_overview(db)
    return StandardResponse(data=PlatformOverviewResponse(**data))


@router.get(
    "/admin/financial",
    response_model=StandardResponse[FinancialOperationsResponse],
    dependencies=[Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))]
)
def get_admin_financial_analytics(db: Session = Depends(get_db)):
    """Provides payment capture rates, payout success, reconciliation exceptions, and ledger balance."""
    data = PlatformAnalyticsService.get_financial_operations_analytics(db)
    return StandardResponse(data=FinancialOperationsResponse(**data))


@router.get(
    "/admin/verification",
    response_model=StandardResponse[ReviewVerificationOperationsResponse],
    dependencies=[Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))]
)
def get_admin_verification_analytics(db: Session = Depends(get_db)):
    """Provides review case turnaround, reviewer workload, and AI verification statistics."""
    data = PlatformAnalyticsService.get_review_and_verification_analytics(db)
    return StandardResponse(data=ReviewVerificationOperationsResponse(**data))


@router.get(
    "/admin/health",
    response_model=StandardResponse[OperationalHealthResponse],
    dependencies=[Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))]
)
def get_admin_operational_health(db: Session = Depends(get_db)):
    """Provides outbox queue status, webhook reliability metrics, and impact ledger integrity."""
    data = PlatformAnalyticsService.get_operational_health(db)
    return StandardResponse(data=OperationalHealthResponse(**data))


# ============================================================================
# Export Endpoints
# ============================================================================

@router.get("/export/ngo/{organization_id}/csv")
def export_ngo_csv(
    organization_id: str = Path(..., description="Organization UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Exports an organization's projects to CSV with formula injection sanitization."""
    verify_ngo_access(organization_id, current_user, db)
    csv_data = AnalyticsExportService.export_ngo_projects_csv(db, organization_id)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ngo_projects_{organization_id[:8]}.csv"}
    )


@router.get("/export/donor/csv")
def export_donor_csv(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Exports a donor's contribution history to CSV with formula injection sanitization."""
    csv_data = AnalyticsExportService.export_donor_contributions_csv(db, current_user.id)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=donor_contributions_{current_user.id[:8]}.csv"}
    )

