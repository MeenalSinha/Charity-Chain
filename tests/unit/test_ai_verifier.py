"""
test_ai_verifier.py - Computer vision & forensic verification engine tests
"""

import pytest
from charity_tracker import EnhancedAIVerifier, Config


def test_error_level_analysis(ai_verifier, sample_jpeg_bytes):
    score, status = ai_verifier.error_level_analysis(sample_jpeg_bytes)
    assert isinstance(score, float)
    assert score >= 0.0
    assert status in ["PASSED", "FAILED"]


def test_perceptual_hash_similarity(ai_verifier, sample_jpeg_bytes, sample_modified_jpeg_bytes):
    # Identical image -> 0 distance
    dist_same, status_same = ai_verifier.perceptual_hash_similarity(sample_jpeg_bytes, sample_jpeg_bytes)
    assert dist_same == 0
    assert status_same == "PASSED"
    
    # Different image -> positive distance
    dist_diff, status_diff = ai_verifier.perceptual_hash_similarity(sample_jpeg_bytes, sample_modified_jpeg_bytes)
    assert dist_diff > 0
    assert status_diff == "PASSED"


def test_gps_distance_haversine(ai_verifier):
    # New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5,570 km
    dist = ai_verifier.calculate_gps_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500000 < dist < 5700000
    
    # Same point -> 0 meters
    dist_zero = ai_verifier.calculate_gps_distance(12.34, 56.78, 12.34, 56.78)
    assert dist_zero == pytest.approx(0.0, abs=0.1)


def test_verify_gps_without_exif_returns_unavailable(ai_verifier, sample_jpeg_bytes):
    # Synthetic image has no EXIF GPS
    passed, msg, dist, status = ai_verifier.verify_gps_location(sample_jpeg_bytes, 10.0, 20.0)
    assert passed is False
    assert status == "UNAVAILABLE"
    assert "No GPS metadata" in msg
    assert dist is None


def test_dms_conversion(ai_verifier):
    # 12 degrees, 30 minutes, 36 seconds = 12.51 degrees
    dms = (12, 30, 36)
    deg = ai_verifier._convert_dms_to_degrees(dms)
    assert deg == pytest.approx(12.51, abs=0.001)


def test_detect_objects_no_fake_fallback(ai_verifier, sample_jpeg_bytes):
    detections, status = ai_verifier.detect_objects(sample_jpeg_bytes)
    assert isinstance(detections, list)
    assert status in ["PASSED", "FAILED", "UNAVAILABLE"]
    # If YOLO weights exist, detections are real; if not, status is UNAVAILABLE and detections empty
    if status == "UNAVAILABLE":
        assert len(detections) == 0


def test_parallel_verification_structured_statuses(ai_verifier, sample_jpeg_bytes, sample_modified_jpeg_bytes):
    # Run full verification pipeline
    result = ai_verifier.run_parallel_verification(
        evidence_bytes=sample_modified_jpeg_bytes,
        before_bytes=sample_jpeg_bytes,
        expected_lat=12.9716,
        expected_lon=77.5946
    )
    
    assert 'timestamp' in result
    assert 'checks' in result
    assert 'confidence' in result
    assert 'pass' in result
    
    checks = result['checks']
    assert 'gps' in checks
    assert 'tamper' in checks
    assert 'object_detection' in checks
    assert 'before_after' in checks
    
    # Validate structured statuses
    for check_name, check_data in checks.items():
        assert check_data['status'] in ['PASSED', 'FAILED', 'UNAVAILABLE', 'NOT_APPLICABLE', 'PENDING']
        assert isinstance(check_data['passed'], bool)
        assert 'message' in check_data
    
    # Check that when before_bytes is provided, before_after is evaluated
    assert checks['before_after']['status'] in ['PASSED', 'FAILED']


def test_parallel_verification_not_applicable_checks(ai_verifier, sample_jpeg_bytes):
    # Run without before_bytes and without expected coordinates
    result = ai_verifier.run_parallel_verification(
        evidence_bytes=sample_jpeg_bytes,
        before_bytes=None,
        expected_lat=None,
        expected_lon=None
    )
    
    checks = result['checks']
    assert checks['before_after']['status'] == 'NOT_APPLICABLE'
    assert checks['before_after']['passed'] is False
    assert checks['gps']['status'] == 'NOT_APPLICABLE'
    
    # Only applicable checks (tamper, object_detection) should determine confidence
    assert 0.0 <= result['confidence'] <= 1.0

