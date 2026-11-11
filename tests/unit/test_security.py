"""
test_security.py - Security validation tests
"""

import time
import pytest
from charity_tracker import SecurityValidator, Config


def test_validate_mime_type_jpeg(sample_jpeg_bytes):
    valid, mime = SecurityValidator.validate_mime_type(sample_jpeg_bytes)
    assert valid is True
    assert mime == 'image/jpeg'


def test_validate_mime_type_png(sample_png_bytes):
    valid, mime = SecurityValidator.validate_mime_type(sample_png_bytes)
    assert valid is True
    assert mime == 'image/png'


def test_validate_mime_type_invalid():
    invalid_bytes = b"NOT_AN_IMAGE_HEADER_AT_ALL"
    valid, mime = SecurityValidator.validate_mime_type(invalid_bytes)
    assert valid is False
    assert mime == 'Unknown'


def test_validate_file_size_ok(sample_jpeg_bytes):
    valid, size_mb = SecurityValidator.validate_file_size(sample_jpeg_bytes)
    assert valid is True
    assert size_mb < Config.MAX_UPLOAD_SIZE_MB


def test_validate_file_size_exceeded():
    large_bytes = b"x" * int((Config.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024)
    valid, size_mb = SecurityValidator.validate_file_size(large_bytes)
    assert valid is False
    assert size_mb > Config.MAX_UPLOAD_SIZE_MB


def test_rate_limiting():
    user = f"test_rate_user_{int(time.time() * 1000)}"
    rate_file = Config.DATA_DIR / f'rate_{user}.json'
    try:
        # Initially under limit
        assert SecurityValidator.check_rate_limit(user) is True
        
        # Record uploads
        for _ in range(5):
            SecurityValidator.record_upload(user)
        
        # Still under limit of 10
        assert SecurityValidator.check_rate_limit(user) is True
    finally:
        if rate_file.exists():
            try:
                rate_file.unlink(missing_ok=True)
            except Exception:
                pass
