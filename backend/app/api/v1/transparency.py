"""
transparency.py - Public Transparency & Verifiable Impact API Endpoints
========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides unauthenticated, privacy-safe endpoints for global platform statistics,
public project transparency pages, NGO profiles, and geographic reach.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, Path, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import StandardResponse
from backend.app.schemas.analytics import (
    PublicTransparencyOverviewResponse, PublicProjectListItem,
    PublicProjectDetailsResponse, PublicOrganizationProfileResponse,
    PublicMapMarkerItem, RegionalDensityResponse, MethodologyResponse
)
from backend.app.db.session import get_db
from backend.app.services.analytics.transparency_service import PublicTransparencyService
from backend.app.services.analytics.geographic_analytics_service import GeographicAnalyticsService

router = APIRouter(prefix="/transparency", tags=["Public Transparency Portal"])


@router.get("/overview", response_model=StandardResponse[PublicTransparencyOverviewResponse])
def get_public_overview(db: Session = Depends(get_db)):
    """Returns global platform metrics safe for public inspection."""
    data = PublicTransparencyService.get_global_overview(db)
    return StandardResponse(data=PublicTransparencyOverviewResponse(**data))


@router.get("/projects", response_model=StandardResponse[List[PublicProjectListItem]])
def list_public_projects(
    category: Optional[str] = Query(None, description="Category filter"),
    status: Optional[str] = Query(None, description="Project status filter"),
    limit: int = Query(50, ge=1, le=100, description="Items limit"),
    offset: int = Query(0, ge=0, description="Items offset"),
    db: Session = Depends(get_db)
):
    """Lists publicly accessible projects with milestone completion and funding progress."""
    data = PublicTransparencyService.get_public_projects(db, category=category, status=status, limit=limit, offset=offset)
    return StandardResponse(data=[PublicProjectListItem(**p) for p in data])


@router.get("/projects/{project_id}", response_model=StandardResponse[PublicProjectDetailsResponse])
def get_public_project_details(
    project_id: str = Path(..., description="Project UUID"),
    db: Session = Depends(get_db)
):
    """Returns detailed public transparency data, milestone status, and certificates for a project."""
    data = PublicTransparencyService.get_public_project_details(db, project_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Public project '{project_id}' not found."
        )
    return StandardResponse(data=PublicProjectDetailsResponse(**data))


@router.get("/organizations/{organization_id}", response_model=StandardResponse[PublicOrganizationProfileResponse])
def get_public_organization_profile(
    organization_id: str = Path(..., description="Organization UUID"),
    db: Session = Depends(get_db)
):
    """Returns public profile, project counts, and verified payouts for an NGO."""
    data = PublicTransparencyService.get_public_organization_profile(db, organization_id)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization '{organization_id}' not found."
        )
    return StandardResponse(data=PublicOrganizationProfileResponse(**data))


@router.get("/geography", response_model=StandardResponse[List[PublicMapMarkerItem]])
def get_public_map_markers(
    category: Optional[str] = Query(None, description="Category filter"),
    status: Optional[str] = Query(None, description="Project status filter"),
    country: Optional[str] = Query(None, description="Country filter"),
    db: Session = Depends(get_db)
):
    """Returns validated, privacy-sanitized map markers for published charitable projects."""
    data = GeographicAnalyticsService.get_public_impact_map_data(db, category=category, status=status, country=country)
    return StandardResponse(data=[PublicMapMarkerItem(**m) for m in data])


@router.get("/geography/density", response_model=StandardResponse[RegionalDensityResponse])
def get_regional_density(db: Session = Depends(get_db)):
    """Returns project density and funding distribution aggregated by country."""
    data = GeographicAnalyticsService.get_regional_density_aggregates(db)
    return StandardResponse(data=RegionalDensityResponse(**data))


@router.get("/methodology", response_model=StandardResponse[MethodologyResponse])
def get_methodology():
    """Returns full explanation of metric calculation rules, data sources, and privacy exclusions."""
    data = PublicTransparencyService.get_methodology()
    return StandardResponse(data=MethodologyResponse(**data))

