"""
test_exif_gps_and_pdf.py - Unit Tests for EXIF, GPS Geospatial, and PDF Processing
==================================================================================
Tests metadata extraction, coordinate conversion, Haversine distance, and bounded PDF handling.
"""

import pytest
from io import BytesIO
from PIL import Image
from pypdf import PdfWriter

from backend.app.core.constants import VerificationCheckStatus
from backend.app.integrations.ai.exif_analyzer import ExifMetadataAnalyzer
from backend.app.integrations.ai.gps_analyzer import GpsGeospatialAnalyzer
from backend.app.integrations.ai.pdf_processor import SafePdfProcessor


def test_exif_synthetic_image_missing_headers(sample_jpeg_bytes):
    """Verify image without EXIF headers returns UNAVAILABLE without error."""
    analyzer = ExifMetadataAnalyzer()
    res = analyzer.extract_metadata(sample_jpeg_bytes)
    assert res.has_exif is False
    assert res.has_gps is False
    assert res.status == VerificationCheckStatus.UNAVAILABLE


def test_gps_dms_conversion():
    """Verify DMS to decimal degrees conversion accuracy."""
    analyzer = GpsGeospatialAnalyzer()
    # 37 degrees, 46 minutes, 29.88 seconds = 37.774967 degrees
    deg = analyzer._convert_dms_to_degrees((37, 46, 29.88))
    assert deg == pytest.approx(37.774967, abs=0.0001)


def test_gps_haversine_distance():
    """Verify Great-Circle Haversine distance calculation."""
    analyzer = GpsGeospatialAnalyzer()
    # New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5,570 km
    dist = analyzer.calculate_haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500000 < dist < 5700000

    # Same location -> 0.0 meters
    zero_dist = analyzer.calculate_haversine_distance(12.9716, 77.5946, 12.9716, 77.5946)
    assert zero_dist == pytest.approx(0.0, abs=0.1)


def test_gps_without_exif_returns_unavailable(sample_jpeg_bytes):
    """Verify that image lacking GPS coordinates returns UNAVAILABLE."""
    analyzer = GpsGeospatialAnalyzer(tolerance_meters=1000.0)
    res = analyzer.verify_location(sample_jpeg_bytes, expected_lat=12.9716, expected_lon=77.5946)
    assert res.status == VerificationCheckStatus.UNAVAILABLE
    assert res.passed is False
    assert res.distance_meters is None
    assert "no gps metadata" in res.message.lower()


def test_pdf_safe_processing(sample_pdf_bytes):
    """Verify safe bounded PDF structural inspection."""
    processor = SafePdfProcessor(max_pages=10)
    res = processor.inspect_pdf(sample_pdf_bytes)
    assert res["status"] == VerificationCheckStatus.PASSED
    assert res["passed"] is True
    assert res["page_count"] == 1


def test_pdf_oversized_pages_flagged():
    """Verify PDF exceeding page limits is safely flagged without exhausting memory."""
    writer = PdfWriter()
    for _ in range(15):
        writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    large_pdf_bytes = buffer.getvalue()

    processor = SafePdfProcessor(max_pages=10)
    res = processor.inspect_pdf(large_pdf_bytes)
    assert res["status"] == VerificationCheckStatus.FLAGGED
    assert res["passed"] is False
    assert res["page_count"] == 15
    assert "exceeding" in res["message"].lower()

