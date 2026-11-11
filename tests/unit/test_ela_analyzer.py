"""
test_ela_analyzer.py - Unit Tests for Error Level Analysis (ELA)
================================================================
Verifies JPEG compression difference calculation, PNG not-applicable handling,
threshold anomaly detection, and corrupt input handling.
"""

import pytest
from backend.app.core.constants import VerificationCheckStatus
from backend.app.integrations.ai.ela_analyzer import ErrorLevelAnalyzer


def test_ela_jpeg_calculation(sample_jpeg_bytes):
    """Verify ELA calculates positive error score for valid JPEG."""
    analyzer = ErrorLevelAnalyzer(threshold=15.0)
    res = analyzer.analyze(sample_jpeg_bytes, mime_type="image/jpeg", generate_artifact=False)
    assert res.is_jpeg is True
    assert res.ela_score >= 0.0
    assert res.status in [VerificationCheckStatus.PASSED, VerificationCheckStatus.FLAGGED]
    assert "compression" in res.message.lower()


def test_ela_png_returns_not_applicable(sample_png_bytes):
    """Verify PNG images return NOT_APPLICABLE because ELA requires lossy compression."""
    analyzer = ErrorLevelAnalyzer()
    res = analyzer.analyze(sample_png_bytes, mime_type="image/png", generate_artifact=False)
    assert res.status == VerificationCheckStatus.NOT_APPLICABLE
    assert res.is_jpeg is False
    assert res.anomaly_detected is False
    assert "lossless" in res.limitations.lower()


def test_ela_anomaly_detection_threshold(sample_jpeg_bytes):
    """Verify that an artificially low threshold flags an anomaly."""
    analyzer = ErrorLevelAnalyzer(threshold=0.0001)
    res = analyzer.analyze(sample_jpeg_bytes, mime_type="image/jpeg", generate_artifact=False)
    assert res.status == VerificationCheckStatus.FLAGGED
    assert res.passed is False
    assert res.anomaly_detected is True
    assert "elevated" in res.message.lower()


def test_ela_corrupt_bytes_error_handling():
    """Verify that malformed bytes produce an ERROR status without raising unhandled exceptions."""
    analyzer = ErrorLevelAnalyzer()
    res = analyzer.analyze(b"\xff\xd8\xff_corrupted_jpeg_data", mime_type="image/jpeg", generate_artifact=False)
    assert res.status == VerificationCheckStatus.ERROR
    assert res.passed is False

