"""
provider.py - Production Multi-Indicator AI & Digital Forensics Provider
========================================================================
Implements AIProviderInterface coordinating YOLOv8 object detection,
Error Level Analysis (ELA), perceptual hashing (pHash), structural similarity (SSIM),
EXIF metadata, Haversine GPS geospatial verification, baseline progression comparison,
PDF document analysis, and grounded XAI explainability summaries.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional

from backend.app.core.config import settings
from backend.app.core.constants import (
    VerificationCheckStatus, OverallAssessmentStatus, VerificationCheckType
)
from backend.app.core.logging import logger
from backend.app.integrations.ai.base import AIProviderInterface
from backend.app.integrations.ai.schemas import (
    SingleCheckResult, VerificationOutcome, BoundingBox
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


class MultiIndicatorAIProvider(AIProviderInterface):
    """Authoritative multi-indicator forensic and computer-vision verification provider."""

    def __init__(self):
        self.yolo = YoloObjectDetector()
        self.ela = ErrorLevelAnalyzer()
        self.phash = PerceptualHashAnalyzer()
        self.ssim = StructuralSimilarityAnalyzer()
        self.exif = ExifMetadataAnalyzer()
        self.gps = GpsGeospatialAnalyzer()
        self.progress = ProgressComparator()
        self.pdf = SafePdfProcessor()
        self.explainer = GroundedLlmExplainer()

    def analyze_evidence_sync(
        self,
        evidence_bytes: bytes,
        mime_type: str,
        expected_sha256: str,
        baseline_bytes: Optional[bytes] = None,
        baseline_evidence_id: Optional[str] = None,
        baseline_hash: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationOutcome:
        """
        Executes complete multi-indicator forensic and visual analysis synchronously.
        """
        ctx = context or {}
        project_title = str(ctx.get("project_title", "Charity Project"))
        milestone_title = str(ctx.get("milestone_title", "Project Milestone"))
        project_desc = str(ctx.get("project_description", ""))
        category_name = str(ctx.get("category_name", ""))
        expected_lat = ctx.get("expected_lat")
        expected_lon = ctx.get("expected_lon")
        generate_artifacts = ctx.get("generate_artifacts", True)

        check_results: List[SingleCheckResult] = []
        artifacts: List[str] = []
        limitations: List[str] = []

        # 1. Cryptographic File Integrity Check
        computed_hash = hashlib.sha256(evidence_bytes).hexdigest()
        hash_matched = (computed_hash.lower() == expected_sha256.lower())
        check_results.append(SingleCheckResult(
            check_type=VerificationCheckType.FILE_INTEGRITY.value,
            status=VerificationCheckStatus.PASSED if hash_matched else VerificationCheckStatus.FAILED,
            passed=hash_matched,
            message="SHA-256 byte-level storage integrity verified." if hash_matched else "CRITICAL: Byte-level SHA-256 checksum mismatch!",
            technical_result={"expected_hash": expected_sha256, "computed_hash": computed_hash},
            raw_metrics={"byte_size": len(evidence_bytes), "hash_match": hash_matched}
        ))

        if not hash_matched:
            return VerificationOutcome(
                pipeline_version=settings.VERIFICATION_PIPELINE_VERSION,
                overall_assessment=OverallAssessmentStatus.INCONSISTENT,
                confidence_score=0.0,
                calibration_status=settings.CALIBRATION_STATUS,
                checks=check_results,
                summary_notes="Integrity failure: Stored file hash does not match recorded ledger checksum.",
                limitations=["Analysis aborted due to storage integrity corruption."],
                model_metadata={"pipeline": settings.VERIFICATION_PIPELINE_VERSION},
                diagnostic_artifacts=[],
                error_details="Cryptographic integrity verification failed."
            )

        # Handle PDF documents
        is_pdf = (mime_type == "application/pdf") or evidence_bytes.startswith(b"%PDF")
        analysis_image_bytes = evidence_bytes

        if is_pdf:
            pdf_result = self.pdf.inspect_pdf(evidence_bytes)
            check_results.append(SingleCheckResult(
                check_type="PDF_DOCUMENT_VALIDATION",
                status=pdf_result["status"],
                passed=pdf_result["passed"],
                message=pdf_result["message"],
                technical_result={"metadata": pdf_result.get("metadata", {}), "page_count": pdf_result.get("page_count", 0)},
                raw_metrics={"page_count": pdf_result.get("page_count", 0)}
            ))
            if pdf_result.get("first_page_image_bytes"):
                analysis_image_bytes = pdf_result["first_page_image_bytes"]
            else:
                limitations.append("PDF document does not contain extractable raster imagery; image-based computer vision checks skipped.")

        # If we have an image (JPEG, PNG, or extracted PDF page image)
        if not is_pdf or analysis_image_bytes != evidence_bytes:
            # 2. EXIF Metadata Extraction
            exif_res = self.exif.extract_metadata(analysis_image_bytes)
            check_results.append(SingleCheckResult(
                check_type=VerificationCheckType.EXIF_METADATA.value,
                status=exif_res.status,
                passed=exif_res.passed,
                message=exif_res.message,
                technical_result={
                    "camera_make": exif_res.camera_make,
                    "camera_model": exif_res.camera_model,
                    "datetime_original": exif_res.datetime_original,
                    "dimensions": f"{exif_res.image_width}x{exif_res.image_height}"
                },
                raw_metrics={"has_exif": exif_res.has_exif, "has_gps": exif_res.has_gps}
            ))

            # 3. GPS Geospatial Proximity
            gps_res = self.gps.verify_location(analysis_image_bytes, expected_lat, expected_lon)
            check_results.append(SingleCheckResult(
                check_type=VerificationCheckType.GPS_CONSISTENCY.value,
                status=gps_res.status,
                passed=gps_res.passed,
                message=gps_res.message,
                technical_result={
                    "extracted_coords": [gps_res.extracted_latitude, gps_res.extracted_longitude] if gps_res.extracted_latitude else None,
                    "expected_coords": [gps_res.expected_latitude, gps_res.expected_longitude] if gps_res.expected_latitude else None,
                    "distance_meters": gps_res.distance_meters,
                    "tolerance_meters": gps_res.tolerance_meters
                },
                raw_metrics={"distance_meters": gps_res.distance_meters}
            ))

            # 4. Error Level Analysis (ELA)
            ela_res = self.ela.analyze(analysis_image_bytes, mime_type, generate_artifact=generate_artifacts)
            if ela_res.diagnostic_artifact:
                artifacts.append(ela_res.diagnostic_artifact)
            check_results.append(SingleCheckResult(
                check_type=VerificationCheckType.ELA_TAMPER.value,
                status=ela_res.status,
                passed=ela_res.passed,
                message=ela_res.message,
                technical_result={
                    "ela_score": ela_res.ela_score,
                    "threshold": ela_res.threshold,
                    "anomaly_detected": ela_res.anomaly_detected
                },
                raw_metrics={"ela_score": ela_res.ela_score},
                diagnostic_artifact_path=ela_res.diagnostic_artifact
            ))
            if ela_res.limitations:
                limitations.append(ela_res.limitations)

            # 5. YOLOv8 Object Detection
            yolo_res = self.yolo.detect_objects(
                analysis_image_bytes,
                category_name=category_name,
                milestone_title=milestone_title,
                project_description=project_desc,
                generate_artifact=generate_artifacts
            )
            if yolo_res.diagnostic_artifact:
                artifacts.append(yolo_res.diagnostic_artifact)
            check_results.append(SingleCheckResult(
                check_type=VerificationCheckType.YOLO_OBJECTS.value,
                status=yolo_res.status,
                passed=yolo_res.passed,
                message=yolo_res.message,
                technical_result={
                    "total_detections": yolo_res.total_detections,
                    "relevant_detections": yolo_res.relevant_detections_count,
                    "top_classes": [d.label for d in yolo_res.detections[:5]],
                    "expected_categories": yolo_res.expected_categories
                },
                raw_metrics={
                    "total": yolo_res.total_detections,
                    "relevant": yolo_res.relevant_detections_count,
                    "duration_ms": yolo_res.inference_duration_ms
                },
                duration_ms=yolo_res.inference_duration_ms,
                model_name=yolo_res.model_name,
                model_version=yolo_res.model_version,
                diagnostic_artifact_path=yolo_res.diagnostic_artifact
            ))

            # 6. Baseline Progress Comparison (if baseline supplied)
            if baseline_bytes:
                prog_res = self.progress.compare_progression(
                    current_bytes=analysis_image_bytes,
                    baseline_bytes=baseline_bytes,
                    baseline_evidence_id=baseline_evidence_id,
                    baseline_hash=baseline_hash,
                    generate_artifact=generate_artifacts
                )
                if prog_res.diagnostic_artifact:
                    artifacts.append(prog_res.diagnostic_artifact)
                check_results.append(SingleCheckResult(
                    check_type=VerificationCheckType.PROGRESS_COMPARISON.value,
                    status=prog_res.status,
                    passed=prog_res.passed,
                    message=prog_res.message,
                    technical_result={
                        "phash_distance": prog_res.phash_distance,
                        "ssim_score": prog_res.ssim_score,
                        "progression_detected": prog_res.progression_detected,
                        "baseline_evidence_id": prog_res.baseline_evidence_id
                    },
                    raw_metrics={
                        "phash_distance": prog_res.phash_distance,
                        "ssim_score": prog_res.ssim_score
                    },
                    diagnostic_artifact_path=prog_res.diagnostic_artifact
                ))
            else:
                check_results.append(SingleCheckResult(
                    check_type=VerificationCheckType.PROGRESS_COMPARISON.value,
                    status=VerificationCheckStatus.NOT_APPLICABLE,
                    passed=True,
                    message="No baseline evidence photograph provided for progression comparison.",
                    technical_result={"reason": "No baseline linked"}
                ))
        else:
            # Document PDF without raster images
            check_results.append(SingleCheckResult(
                check_type=VerificationCheckType.YOLO_OBJECTS.value,
                status=VerificationCheckStatus.NOT_APPLICABLE,
                passed=True,
                message="Visual object detection is not applicable to non-raster PDF documents."
            ))

        # Overall Multi-Indicator Assessment & Heuristic Confidence Aggregation
        overall_status, confidence_score = self._aggregate_findings(check_results)

        # 7. Grounded LLM / Rule-Based Explainability
        expl_res = self.explainer.generate_explanation(
            checks=check_results,
            overall_assessment=overall_status.value,
            project_title=project_title,
            milestone_title=milestone_title
        )
        check_results.append(SingleCheckResult(
            check_type=VerificationCheckType.LLM_EXPLAINABILITY.value,
            status=expl_res.status,
            passed=expl_res.passed,
            message=expl_res.summary_text,
            technical_result={
                "key_findings": expl_res.key_findings,
                "anomalies_flagged": expl_res.anomalies_flagged,
                "reviewer_questions": expl_res.recommended_reviewer_questions,
                "provider": expl_res.provider_name
            },
            model_name=expl_res.model_name
        ))

        limitations.extend([
            "AI verification findings are advisory indicators for human reviewers; they do not trigger fund disbursement or automatic milestone approval.",
            "YOLO COCO detections represent visual recognition, not verification of real-world charitable impact.",
            "Confidence score is an unvalidated heuristic and does not represent a calibrated probability."
        ])

        return VerificationOutcome(
            pipeline_version=settings.VERIFICATION_PIPELINE_VERSION,
            overall_assessment=overall_status,
            confidence_score=confidence_score,
            calibration_status=settings.CALIBRATION_STATUS,
            checks=check_results,
            summary_notes=expl_res.summary_text,
            limitations=list(set(limitations)),
            model_metadata={
                "pipeline_version": settings.VERIFICATION_PIPELINE_VERSION,
                "yolo_model": self.yolo.model_name,
                "yolo_version": self.yolo.model_version,
                "yolo_weights_hash": self.yolo.weights_hash,
                "explainer_provider": expl_res.provider_name
            },
            diagnostic_artifacts=artifacts,
            error_details=None
        )

    def _aggregate_findings(self, checks: List[SingleCheckResult]) -> Tuple[OverallAssessmentStatus, float]:
        """
        Synthesizes individual check statuses into an overall assessment status and
        calculates a normalized, unvalidated heuristic confidence score.
        """
        weights = {
            VerificationCheckType.FILE_INTEGRITY.value: 0.30,
            VerificationCheckType.YOLO_OBJECTS.value: 0.25,
            VerificationCheckType.ELA_TAMPER.value: 0.20,
            VerificationCheckType.GPS_CONSISTENCY.value: 0.15,
            VerificationCheckType.PROGRESS_COMPARISON.value: 0.10
        }

        any_flagged = any(c.status == VerificationCheckStatus.FLAGGED for c in checks)
        any_failed = any(c.status == VerificationCheckStatus.FAILED for c in checks)
        all_passed = all(c.status in (VerificationCheckStatus.PASSED, VerificationCheckStatus.NOT_APPLICABLE) for c in checks)

        total_active_weight = 0.0
        earned_weight = 0.0

        for c in checks:
            w = weights.get(c.check_type, 0.0)
            if w > 0 and c.status in (VerificationCheckStatus.PASSED, VerificationCheckStatus.FLAGGED, VerificationCheckStatus.FAILED):
                total_active_weight += w
                if c.passed and c.status == VerificationCheckStatus.PASSED:
                    earned_weight += w
                elif c.status == VerificationCheckStatus.FLAGGED:
                    earned_weight += (w * 0.5)

        raw_score = (earned_weight / total_active_weight) if total_active_weight > 0 else 0.5
        confidence = round(max(0.0, min(1.0, raw_score)), 3)

        if any_failed:
            overall = OverallAssessmentStatus.INCONSISTENT
        elif any_flagged:
            overall = OverallAssessmentStatus.NEEDS_FURTHER_REVIEW
        elif all_passed:
            overall = OverallAssessmentStatus.CONSISTENT
        else:
            overall = OverallAssessmentStatus.INSUFFICIENT_DATA

        return overall, confidence

    async def analyze_evidence(
        self,
        image_bytes: bytes,
        baseline_bytes: Optional[bytes] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async interface adhering to base AIProviderInterface."""
        outcome = self.analyze_evidence_sync(
            evidence_bytes=image_bytes,
            mime_type=(context or {}).get("mime_type", "image/jpeg"),
            expected_sha256=(context or {}).get("expected_sha256", hashlib.sha256(image_bytes).hexdigest()),
            baseline_bytes=baseline_bytes,
            context=context
        )
        return outcome.dict()

    async def generate_findings_summary(
        self,
        verification_report: Dict[str, Any],
        project_details: Dict[str, Any]
    ) -> str:
        return verification_report.get("summary_notes", "Summary generated.")

    async def explain_verification_outcome(
        self,
        checks: Dict[str, Any],
        confidence_score: float
    ) -> Dict[str, Any]:
        return {
            "confidence_score": confidence_score,
            "calibration_status": settings.CALIBRATION_STATUS,
            "disclaimer": "AI findings are advisory indicators for human review."
        }

