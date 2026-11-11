"""
__init__.py - AI Verification and Digital Forensics Package
==========================================================
Exports AI provider interfaces, concrete multi-indicator engines,
specialized analyzers, schemas, and provider factories.
"""

from backend.app.integrations.ai.base import AIProviderInterface
from backend.app.integrations.ai.schemas import (
    SingleCheckResult, VerificationOutcome, BoundingBox, ObjectDetection,
    YoloCheckResult, ElaCheckResult, PhashCheckResult, SsimCheckResult,
    ExifCheckResult, GpsCheckResult, ProgressCheckResult, LlmExplanationResult
)
from backend.app.integrations.ai.yolo_provider import YoloObjectDetector
from backend.app.integrations.ai.ela_analyzer import ErrorLevelAnalyzer
from backend.app.integrations.ai.phash_analyzer import PerceptualHashAnalyzer
from backend.app.integrations.ai.ssim_analyzer import StructuralSimilarityAnalyzer
from backend.app.integrations.ai.exif_analyzer import ExifMetadataAnalyzer
from backend.app.integrations.ai.gps_analyzer import GpsGeospatialAnalyzer
from backend.app.integrations.ai.progress_comparator import ProgressComparator
from backend.app.integrations.ai.pdf_processor import SafePdfProcessor
from backend.app.integrations.ai.llm_explainer import GroundedLlmExplainer
from backend.app.integrations.ai.provider import MultiIndicatorAIProvider
from backend.app.integrations.ai.factory import get_ai_provider, set_ai_provider, reset_ai_provider

__all__ = [
    "AIProviderInterface",
    "MultiIndicatorAIProvider",
    "get_ai_provider",
    "set_ai_provider",
    "reset_ai_provider",
    "SingleCheckResult",
    "VerificationOutcome",
    "BoundingBox",
    "ObjectDetection",
    "YoloCheckResult",
    "ElaCheckResult",
    "PhashCheckResult",
    "SsimCheckResult",
    "ExifCheckResult",
    "GpsCheckResult",
    "ProgressCheckResult",
    "LlmExplanationResult",
    "YoloObjectDetector",
    "ErrorLevelAnalyzer",
    "PerceptualHashAnalyzer",
    "StructuralSimilarityAnalyzer",
    "ExifMetadataAnalyzer",
    "GpsGeospatialAnalyzer",
    "ProgressComparator",
    "SafePdfProcessor",
    "GroundedLlmExplainer"
]

