"""
test_geographic_aggregation.py - Geographic Privacy & Coordinate Validation Tests
=================================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import pytest
from backend.app.services.analytics.geographic_analytics_service import GeographicAnalyticsService


def test_coordinate_validation():
    """Verifies latitude and longitude range validation."""
    assert GeographicAnalyticsService.validate_coordinates(40.7128, -74.0060) is True
    assert GeographicAnalyticsService.validate_coordinates(0.0, 0.0) is True
    assert GeographicAnalyticsService.validate_coordinates(-90.0, 180.0) is True

    # Out of range
    assert GeographicAnalyticsService.validate_coordinates(91.0, 0.0) is False
    assert GeographicAnalyticsService.validate_coordinates(0.0, -181.0) is False
    assert GeographicAnalyticsService.validate_coordinates(None, 10.0) is False
    assert GeographicAnalyticsService.validate_coordinates(10.0, None) is False
    assert GeographicAnalyticsService.validate_coordinates("invalid", 10.0) is False


def test_location_privacy_obfuscation():
    """Verifies that sensitive projects have their exact coordinates obfuscated to a regional centroid."""
    exact_lat = 37.774929
    exact_lon = -122.419416

    # Non-sensitive project retains precision
    public_coords = GeographicAnalyticsService.apply_location_privacy(exact_lat, exact_lon, is_sensitive=False)
    assert public_coords["is_obfuscated"] is False
    assert public_coords["latitude"] == round(exact_lat, 5)
    assert public_coords["longitude"] == round(exact_lon, 5)

    # Sensitive project is rounded to coarse regional grid centroid
    sensitive_coords = GeographicAnalyticsService.apply_location_privacy(exact_lat, exact_lon, is_sensitive=True)
    assert sensitive_coords["is_obfuscated"] is True
    assert "privacy_note" in sensitive_coords
    # Coarse coordinate should differ from exact coordinate beyond 3 decimal places
    assert sensitive_coords["latitude"] != exact_lat
    assert sensitive_coords["longitude"] != exact_lon

