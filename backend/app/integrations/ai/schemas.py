"""
schemas.py - Strongly Typed AI Verification & Forensics Data Contracts
======================================================================
Defines strongly validated request, result, and aggregate schemas
for computer vision, image forensics, geospatial consistency, and LLM explainability.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from backend.app.core.constants import VerificationCheckStatus, OverallAssessmentStatus


class BoundingBox(BaseModel):
    """Normalized or pixel coordinates for a detected object."""
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetection(BaseModel):
    """Individual object detection item from YOLOv8."""
    class_id: int
    label: str
    confidence: float
    box: List[float]
    is_project_relevant: bool = False


class YoloCheckResult(BaseModel):
    """Result data for YOLOv8 object detection check."""
    status: VerificationCheckStatus
    passed: bool
    total_detections: int
    relevant_detections_count: int
    detections: List[ObjectDetection]
    model_name: str
    model_version: str
    inference_duration_ms: int
    expected_categories: List[str] = Field(default_factory=list)
    message: str
    diagnostic_artifact: Optional[str] = None


class ElaCheckResult(BaseModel):
    """Result data for Error Level Analysis compression variance check."""
    status: VerificationCheckStatus
    passed: bool
    ela_score: float
    threshold: float
    is_jpeg: bool
    anomaly_detected: bool
    diagnostic_artifact: Optional[str] = None
    message: str
    limitations: str


class PhashCheckResult(BaseModel):
    """Result data for Perceptual Hash (pHash) similarity."""
    status: VerificationCheckStatus
    passed: bool
    hamming_distance: Optional[int] = None
    threshold: int
    similarity_interpretation: str
    message: str


class SsimCheckResult(BaseModel):
    """Result data for Structural Similarity Index (SSIM) image comparison."""
    status: VerificationCheckStatus
    passed: bool
    ssim_score: Optional[float] = None
    threshold: float
    comparison_target: str
    message: str


class ExifCheckResult(BaseModel):
    """Result data for EXIF metadata inspection."""
    status: VerificationCheckStatus
    passed: bool
    has_exif: bool
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    datetime_original: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    has_gps: bool = False
    message: str


class GpsCheckResult(BaseModel):
    """Result data for GPS and geospatial consistency check."""
    status: VerificationCheckStatus
    passed: bool
    extracted_latitude: Optional[float] = None
    extracted_longitude: Optional[float] = None
    expected_latitude: Optional[float] = None
    expected_longitude: Optional[float] = None
    distance_meters: Optional[float] = None
    tolerance_meters: float
    message: str


class ProgressCheckResult(BaseModel):
    """Result data for milestone progress comparison against baseline evidence."""
    status: VerificationCheckStatus
    passed: bool
    baseline_evidence_id: Optional[str] = None
    baseline_hash: Optional[str] = None
    phash_distance: Optional[int] = None
    ssim_score: Optional[float] = None
    progression_detected: bool
    message: str
    diagnostic_artifact: Optional[str] = None


class LlmExplanationResult(BaseModel):
    """Result data for optional LLM-generated explainable verification summary."""
    status: VerificationCheckStatus
    passed: bool
    provider_name: str
    model_name: str
    summary_text: str
    key_findings: List[str] = Field(default_factory=list)
    anomalies_flagged: List[str] = Field(default_factory=list)
    recommended_reviewer_questions: List[str] = Field(default_factory=list)
    prompt_injection_detected: bool = False
    message: str


class SingleCheckResult(BaseModel):
    """Generic envelope for an executed verification check."""
    check_type: str
    status: VerificationCheckStatus
    passed: bool
    message: str
    technical_result: Dict[str, Any] = Field(default_factory=dict)
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    diagnostic_artifact_path: Optional[str] = None
    error_message: Optional[str] = None


class VerificationOutcome(BaseModel):
    """Aggregate multi-indicator verification result."""
    pipeline_version: str
    overall_assessment: OverallAssessmentStatus
    confidence_score: Optional[float] = None
    calibration_status: str
    checks: List[SingleCheckResult]
    summary_notes: str
    limitations: List[str]
    model_metadata: Dict[str, Any]
    diagnostic_artifacts: List[str] = Field(default_factory=list)
    error_details: Optional[str] = None

