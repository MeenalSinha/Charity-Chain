"""
test_yolo_detector.py - Unit Tests for YOLOv8 Object Detection Module
=====================================================================
Tests YOLO model loading, hash verification, missing model handling,
category relevance mapping, and corrupt input resilience.
"""

import pytest
from backend.app.core.constants import VerificationCheckStatus
from backend.app.integrations.ai.yolo_provider import YoloObjectDetector
from backend.app.integrations.ai.relevance_mapper import resolve_expected_classes


def test_yolo_model_initialization_and_weights_hash():
    """Verify YOLO detector initializes model and records weights SHA-256 hash."""
    detector = YoloObjectDetector(model_path="yolov8n.pt")
    assert detector.is_loaded is True
    assert detector.weights_hash is not None
    assert len(detector.weights_hash) == 64
    assert detector.model_name == "YOLOv8n"


def test_yolo_inference_on_synthetic_image(sample_jpeg_bytes):
    """Verify genuine YOLO inference execution without fabricating detections."""
    detector = YoloObjectDetector(model_path="yolov8n.pt")
    result = detector.detect_objects(
        image_bytes=sample_jpeg_bytes,
        category_name="Water & Sanitation",
        milestone_title="Solar water pump installation"
    )
    assert result.status in [VerificationCheckStatus.PASSED, VerificationCheckStatus.FAILED]
    assert isinstance(result.total_detections, int)
    assert isinstance(result.detections, list)
    assert result.inference_duration_ms >= 0
    assert result.model_name == "YOLOv8n"


def test_yolo_missing_model_returns_unavailable(sample_jpeg_bytes):
    """Verify that a missing model file returns UNAVAILABLE without fabricating detections."""
    detector = YoloObjectDetector(model_path="non_existent_yolo_weights.pt")
    assert detector.is_loaded is False
    result = detector.detect_objects(sample_jpeg_bytes)
    assert result.status == VerificationCheckStatus.UNAVAILABLE
    assert result.passed is False
    assert result.total_detections == 0
    assert len(result.detections) == 0
    assert "unavailable" in result.message.lower()


def test_yolo_category_relevance_annotation():
    """Verify category keywords are resolved to expected YOLO class names."""
    water_classes = resolve_expected_classes(category_name="Water & Sanitation")
    assert "bottle" in water_classes or "sink" in water_classes

    edu_classes = resolve_expected_classes(category_name="Education")
    assert "book" in edu_classes or "chair" in edu_classes

    env_classes = resolve_expected_classes(milestone_title="Tree planting in forest reserve")
    assert "potted plant" in env_classes or "bird" in env_classes


def test_yolo_corrupt_image_handling():
    """Verify corrupt image bytes return an explicit ERROR status."""
    detector = YoloObjectDetector(model_path="yolov8n.pt")
    corrupt_bytes = b"NOT_A_VALID_JPEG_OR_IMAGE_DATA"
    result = detector.detect_objects(corrupt_bytes)
    assert result.status == VerificationCheckStatus.ERROR
    assert result.passed is False
    assert "failed" in result.message.lower()

