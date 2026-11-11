"""
test_phash_and_ssim.py - Unit Tests for pHash, SSIM, and Progress Comparator
=============================================================================
Tests perceptual hashing, Structural Similarity Index, and before/after comparison.
"""

import pytest
from backend.app.core.constants import VerificationCheckStatus
from backend.app.integrations.ai.phash_analyzer import PerceptualHashAnalyzer
from backend.app.integrations.ai.ssim_analyzer import StructuralSimilarityAnalyzer
from backend.app.integrations.ai.progress_comparator import ProgressComparator


def test_phash_identical_images(sample_jpeg_bytes):
    """Verify that identical images produce distance 0 and IDENTICAL_IMAGE interpretation."""
    analyzer = PerceptualHashAnalyzer()
    res = analyzer.compare_images(sample_jpeg_bytes, sample_jpeg_bytes)
    assert res.status == VerificationCheckStatus.PASSED
    assert res.hamming_distance == 0
    assert res.similarity_interpretation == "IDENTICAL_IMAGE"


def test_phash_different_images(sample_jpeg_bytes, sample_modified_jpeg_bytes):
    """Verify that visually different images produce distance > 0."""
    analyzer = PerceptualHashAnalyzer()
    res = analyzer.compare_images(sample_jpeg_bytes, sample_modified_jpeg_bytes)
    assert res.status == VerificationCheckStatus.PASSED
    assert res.hamming_distance > 0


def test_ssim_identical_images(sample_jpeg_bytes):
    """Verify that identical images produce SSIM score close to 1.0."""
    analyzer = StructuralSimilarityAnalyzer()
    res = analyzer.compare(sample_jpeg_bytes, sample_jpeg_bytes)
    assert res.status == VerificationCheckStatus.PASSED
    assert res.ssim_score is not None
    assert res.ssim_score >= 0.99


def test_ssim_different_images(sample_jpeg_bytes, sample_modified_jpeg_bytes):
    """Verify that distinct images yield lower SSIM score."""
    analyzer = StructuralSimilarityAnalyzer()
    res = analyzer.compare(sample_jpeg_bytes, sample_modified_jpeg_bytes)
    assert res.status == VerificationCheckStatus.PASSED
    assert res.ssim_score is not None
    assert res.ssim_score < 0.95


def test_progress_comparator_duplicate_flagged(sample_jpeg_bytes):
    """Verify that uploading identical image as baseline is flagged as duplicate."""
    comp = ProgressComparator()
    res = comp.compare_progression(
        current_bytes=sample_jpeg_bytes,
        baseline_bytes=sample_jpeg_bytes,
        baseline_evidence_id="base-1",
        generate_artifact=False
    )
    assert res.status == VerificationCheckStatus.FLAGGED
    assert res.progression_detected is False
    assert "duplicate" in res.message.lower()


def test_progress_comparator_real_progression(sample_jpeg_bytes, sample_modified_jpeg_bytes):
    """Verify that visual evolution is recognized as genuine progression."""
    comp = ProgressComparator()
    res = comp.compare_progression(
        current_bytes=sample_modified_jpeg_bytes,
        baseline_bytes=sample_jpeg_bytes,
        baseline_evidence_id="base-1",
        generate_artifact=False
    )
    assert res.status == VerificationCheckStatus.PASSED
    assert res.progression_detected is True
    assert "progression" in res.message.lower()

