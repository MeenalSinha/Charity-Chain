"""
test_geographic_privacy.py - Location Privacy & Coordinate Obfuscation Security Tests
=====================================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project, ProjectVisibility
from backend.app.core.constants import PlatformRole, AccountStatus, ProjectStatus
from backend.app.services.analytics.geographic_analytics_service import GeographicAnalyticsService


def test_sensitive_shelter_location_is_obfuscated_in_public_markers(db_session):
    """Verifies that sensitive projects (e.g. domestic shelters) do NOT expose exact coordinates to public maps."""
    user = User(
        id=str(uuid.uuid4()),
        email="shelter_admin@example.org",
        full_name="Shelter Admin",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(id=str(uuid.uuid4()), name="Safe Harbor Shelter", creator_id=user.id)

    # Sensitive project: location must NOT leak exact residential coordinates
    exact_lat = 34.052234
    exact_lon = -118.243685

    shelter_proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Emergency Family Shelter",
        description="Confidential refuge facility",
        public_visibility=ProjectVisibility.PUBLIC,
        status=ProjectStatus.ACTIVE,
        latitude=exact_lat,
        longitude=exact_lon,
        is_location_sensitive=True,
        target_amount_minor_units=500000,
        currency="USD"
    )
    db_session.add_all([user, org, shelter_proj])
    db_session.commit()

    markers = GeographicAnalyticsService.get_public_impact_map_data(db_session)
    shelter_marker = next((m for m in markers if m["project_id"] == shelter_proj.id), None)

    assert shelter_marker is not None
    assert shelter_marker["is_obfuscated"] is True
    # Coordinated must be coarsened/obfuscated
    assert shelter_marker["latitude"] != exact_lat
    assert shelter_marker["longitude"] != exact_lon
