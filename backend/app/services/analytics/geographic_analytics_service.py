"""
geographic_analytics_service.py - Geospatial Impact Analytics & Privacy Aggregation
===================================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides validation, regional density aggregation, and privacy-preserving coordinate
obfuscation for sensitive charitable project locations (shelters, medical outposts).
"""

from typing import List, Dict, Any, Optional
import math
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.db.models.projects import Project, ProjectVisibility
from backend.app.core.constants import ProjectStatus


class GeographicAnalyticsService:
    """Processes project coordinates, builds density heatmaps, and enforces location privacy."""

    @staticmethod
    def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
        """Validates that latitude is in [-90, 90] and longitude is in [-180, 180]."""
        if lat is None or lon is None:
            return False
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            if math.isnan(lat_f) or math.isnan(lon_f):
                return False
            return (-90.0 <= lat_f <= 90.0) and (-180.0 <= lon_f <= 180.0)
        except (ValueError, TypeError):
            return False

    @classmethod
    def apply_location_privacy(
        cls,
        lat: float,
        lon: float,
        is_sensitive: bool = False,
        coarse_resolution_degrees: float = 0.15  # ~16 km grid resolution
    ) -> Dict[str, Any]:
        """
        Enforces privacy protections. Sensitive locations (such as domestic violence shelters,
        child welfare centers, or confidential medical sites) are snapped to a coarse regional grid
        centroid, preventing exact address reverse-geocoding.
        """
        if not is_sensitive:
            return {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "is_obfuscated": False
            }

        # Snap to coarse centroid grid
        coarse_lat = round(round(lat / coarse_resolution_degrees) * coarse_resolution_degrees, 3)
        coarse_lon = round(round(lon / coarse_resolution_degrees) * coarse_resolution_degrees, 3)

        return {
            "latitude": coarse_lat,
            "longitude": coarse_lon,
            "is_obfuscated": True,
            "privacy_note": "Coordinates aggregated to regional centroid for beneficiary safety."
        }

    @classmethod
    def get_public_impact_map_data(
        cls,
        db: Session,
        category: Optional[str] = None,
        status: Optional[str] = None,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts validated, privacy-sanitized map pins for public presentation.
        Filters out private projects and handles missing coordinates safely.
        """
        query = select(Project).where(Project.public_visibility == ProjectVisibility.PUBLIC)

        if status:
            query = query.where(Project.status == status)
        if country:
            query = query.where(Project.country.ilike(country))

        projects = db.scalars(query).all()

        markers = []
        for p in projects:
            lat = p.latitude
            lon = p.longitude

            # Safely skip unpositioned or invalid projects without failing
            if not cls.validate_coordinates(lat, lon):
                continue

            # Check if project or category implies sensitivity
            cat_name = p.category.name if p.category else "General"
            if category and category.lower() not in cat_name.lower():
                continue

            is_sensitive = getattr(p, "is_location_sensitive", False)

            # Apply location privacy
            coords = cls.apply_location_privacy(lat, lon, is_sensitive=is_sensitive)

            markers.append({
                "project_id": p.id,
                "title": p.title,
                "organization_name": p.organization.name if p.organization else "Verified NGO",
                "category": cat_name,
                "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "is_obfuscated": coords["is_obfuscated"],
                "target_amount_cents": p.target_amount_minor_units,
                "raised_amount_cents": p.raised_amount_minor_units,
                "target_formatted": f"${p.target_amount_minor_units / 100:,.2f}",
                "raised_formatted": f"${p.raised_amount_minor_units / 100:,.2f}",
                "country": p.country or "Global",
                "city": p.city or "Region"
            })

        return markers

    @classmethod
    def get_regional_density_aggregates(cls, db: Session) -> Dict[str, Any]:
        """Aggregates project counts and funding density by country and region."""
        projects = db.scalars(
            select(Project).where(Project.public_visibility == ProjectVisibility.PUBLIC)
        ).all()

        by_country: Dict[str, Dict[str, Any]] = {}

        for p in projects:
            c = p.country or "Global"
            if c not in by_country:
                by_country[c] = {
                    "country": c,
                    "project_count": 0,
                    "total_raised_cents": 0,
                    "active_projects": 0,
                    "completed_projects": 0
                }
            by_country[c]["project_count"] += 1
            by_country[c]["total_raised_cents"] += p.raised_amount_minor_units
            if p.status == ProjectStatus.ACTIVE:
                by_country[c]["active_projects"] += 1
            elif p.status == ProjectStatus.COMPLETED:
                by_country[c]["completed_projects"] += 1

        density_list = list(by_country.values())
        density_list.sort(key=lambda x: x["project_count"], reverse=True)

        return {
            "total_countries": len(by_country),
            "density": density_list
        }

