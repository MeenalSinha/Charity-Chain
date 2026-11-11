"""
test_input_hardening.py - Input hardening and corrupted file handling tests
"""

import pytest
from charity_tracker import EnhancedAIVerifier, SecurityValidator


def test_corrupted_file_handling(ai_verifier):
    corrupt_bytes = b"CORRUPTED_NON_IMAGE_DATA_BYTES_12345"
    
    # ELA error handling
    score, status = ai_verifier.error_level_analysis(corrupt_bytes)
    assert status == "FAILED"
    assert score == 0.0
    
    # pHash error handling
    dist, phash_status = ai_verifier.perceptual_hash_similarity(corrupt_bytes, corrupt_bytes)
    assert phash_status == "FAILED"
    assert dist is None
    
    # EXIF extraction on invalid bytes
    exif = ai_verifier.get_exif_data(corrupt_bytes)
    assert exif == {}
    
    # GPS extraction on invalid bytes
    coords = ai_verifier.extract_exif_gps(corrupt_bytes)
    assert coords is None


def test_mime_validator_rejects_executable_and_scripts():
    exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
    valid, mime = SecurityValidator.validate_mime_type(exe_bytes)
    assert valid is False
    assert mime == 'Unknown'
    
    html_bytes = b"<html><script>alert(1)</script></html>"
    valid, mime = SecurityValidator.validate_mime_type(html_bytes)
    assert valid is False

