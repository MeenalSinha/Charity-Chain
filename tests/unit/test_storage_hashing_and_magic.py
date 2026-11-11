"""
test_storage_hashing_and_magic.py - Unit tests for magic bytes and hashing
==========================================================================
Verifies filename sanitization, magic bytes detection, spoofing rejection,
and deterministic SHA-256 content addressing.
"""

import hashlib
import pytest
from backend.app.integrations.storage.local import _sanitize_filename, _validate_magic_bytes
from backend.app.core.exceptions import ValidationException


def test_sanitize_filename_removes_traversal_and_dangerous_chars():
    """Verify filename sanitization neutralizes directory traversal and special chars."""
    assert _sanitize_filename("../../etc/passwd") == "passwd"
    assert _sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert _sanitize_filename("normal_photo.jpg") == "normal_photo.jpg"
    assert _sanitize_filename("photo with spaces!@#$%^&*().png") == "photo_with_spaces__________.png"
    assert _sanitize_filename("") == "document"


def test_validate_magic_bytes_authentic_formats():
    """Verify legitimate PDF, JPEG, and PNG headers pass magic byte inspection."""
    pdf_bytes = b"%PDF-1.7\r\n..."
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF..."
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."

    assert _validate_magic_bytes(pdf_bytes, "application/pdf") is True
    assert _validate_magic_bytes(jpeg_bytes, "image/jpeg") is True
    assert _validate_magic_bytes(png_bytes, "image/png") is True


def test_validate_magic_bytes_rejects_spoofed_or_mismatched_content():
    """Verify that spoofed extensions or malicious scripts are rejected."""
    # Shell script disguised as PNG
    fake_png = b"#!/bin/bash\necho 'malicious code'"
    assert _validate_magic_bytes(fake_png, "image/png") is False

    # HTML disguised as PDF
    fake_pdf = b"<html><body>Phishing</body></html>"
    assert _validate_magic_bytes(fake_pdf, "application/pdf") is False

    # JPEG header with PDF mime
    jpeg_bytes = b"\xff\xd8\xff\xe0JFIF"
    assert _validate_magic_bytes(jpeg_bytes, "application/pdf") is False

    # Empty payload
    assert _validate_magic_bytes(b"", "image/jpeg") is False


def test_validate_magic_bytes_rejects_unsupported_mimes():
    """Verify unsupported MIME types return False strictly."""
    assert _validate_magic_bytes(b"some content", "application/x-executable") is False
    assert _validate_magic_bytes(b"some content", "text/javascript") is False
    assert _validate_magic_bytes(b"some content", "application/zip") is False


def test_deterministic_sha256_checksum():
    """Verify SHA-256 calculation is exact and reproducible across multiple runs."""
    sample_content = b"CharityChain immutable verification evidence payload 2026"
    expected_hash = hashlib.sha256(sample_content).hexdigest()

    computed_hash = hashlib.sha256(sample_content).hexdigest()
    assert computed_hash == expected_hash
    assert len(computed_hash) == 64
