"""
ela_analyzer.py - Error Level Analysis (ELA) Digital Forensics Analyzer
=======================================================================
Performs Error Level Analysis (ELA) on lossy JPEG images to detect localized
compression variance anomalies. Generates diagnostic heatmaps and strictly documents
limitations regarding PNG images and editing interpretation.
"""

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import ElaCheckResult


class ErrorLevelAnalyzer:
    """Forensic analyzer computing compression variance across JPEG images."""

    def __init__(self, threshold: Optional[float] = None, recompress_quality: int = 90):
        self.threshold = threshold if threshold is not None else settings.TAMPER_ELA_THRESHOLD
        self.recompress_quality = recompress_quality

    def analyze(
        self,
        image_bytes: bytes,
        mime_type: Optional[str] = None,
        generate_artifact: bool = True
    ) -> ElaCheckResult:
        """
        Calculates ELA metrics and evaluates against configured threshold.
        Gracefully handles non-JPEG formats where ELA is not physically meaningful.
        """
        # Determine if JPEG
        is_jpeg = False
        if mime_type:
            is_jpeg = "jpeg" in mime_type.lower() or "jpg" in mime_type.lower()
        else:
            is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")

        if not is_jpeg:
            return ElaCheckResult(
                status=VerificationCheckStatus.NOT_APPLICABLE,
                passed=True,
                ela_score=0.0,
                threshold=self.threshold,
                is_jpeg=False,
                anomaly_detected=False,
                diagnostic_artifact=None,
                message="Error Level Analysis (ELA) is specific to lossy JPEG compression and is not applicable to lossless formats (e.g. PNG, PDF).",
                limitations="Lossless images do not store DCT quantization tables or JPEG compression artifacts. Check recorded as NOT_APPLICABLE."
            )

        try:
            orig = Image.open(BytesIO(image_bytes))
            # Decompression bomb defense
            if orig.width * orig.height > settings.MAX_IMAGE_PIXELS:
                return ElaCheckResult(
                    status=VerificationCheckStatus.ERROR,
                    passed=False,
                    ela_score=0.0,
                    threshold=self.threshold,
                    is_jpeg=True,
                    anomaly_detected=False,
                    diagnostic_artifact=None,
                    message="Image exceeds maximum permitted pixel dimensions.",
                    limitations="Resource protection limit triggered."
                )
            orig = orig.convert("RGB")

            # Controlled recompression at fixed quality
            buffer = BytesIO()
            orig.save(buffer, "JPEG", quality=self.recompress_quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            # Compute pixel-wise absolute difference
            diff = ImageChops.difference(orig, recompressed)
            diff_arr = np.asarray(diff, dtype=np.float32)

            # Calculate error statistics
            mean_error = float(diff_arr.mean())
            max_error = float(diff_arr.max())
            std_error = float(diff_arr.std())

            anomaly_detected = mean_error >= self.threshold
            status = VerificationCheckStatus.FLAGGED if anomaly_detected else VerificationCheckStatus.PASSED
            passed = not anomaly_detected

            if anomaly_detected:
                msg = f"Elevated compression variance detected (mean error: {mean_error:.2f} >= threshold {self.threshold:.2f}, max: {max_error:.1f}). Possible digital manipulation, multi-generation re-encoding, or spliced elements."
            else:
                msg = f"Compression error variance within expected baseline (mean error: {mean_error:.2f} < threshold {self.threshold:.2f})."

            artifact_path = None
            if generate_artifact:
                artifact_path = self._save_diagnostic_heatmap(diff, mean_error)

            return ElaCheckResult(
                status=status,
                passed=passed,
                ela_score=round(mean_error, 2),
                threshold=self.threshold,
                is_jpeg=True,
                anomaly_detected=anomaly_detected,
                diagnostic_artifact=artifact_path,
                message=msg,
                limitations="ELA identifies compression inconsistencies. Variations can naturally occur from social media re-saving, camera firmware sharpening, or normal editing. It is an investigative indicator, not conclusive proof of fraud."
            )

        except Exception as e:
            logger.error(f"Error Level Analysis failed: {e}")
            return ElaCheckResult(
                status=VerificationCheckStatus.ERROR,
                passed=False,
                ela_score=0.0,
                threshold=self.threshold,
                is_jpeg=True,
                anomaly_detected=False,
                diagnostic_artifact=None,
                message=f"ELA analysis processing error: {e}",
                limitations="Check failed to complete due to processing error."
            )

    def _save_diagnostic_heatmap(self, diff_img: Image.Image, score: float) -> Optional[str]:
        """Amplifies differences and persists a diagnostic visualization."""
        try:
            art_dir = Path(settings.DIAGNOSTIC_ARTIFACTS_DIR)
            art_dir.mkdir(parents=True, exist_ok=True)

            # Boost contrast to make compression differences visible to human reviewers
            enhancer = ImageEnhance.Brightness(diff_img)
            amplified = enhancer.enhance(10.0)

            filename = f"ela_diff_{int(time.time() * 1000)}_{os.getpid()}.jpg"
            dest = art_dir / filename
            amplified.save(dest, format="JPEG", quality=85)
            return str(dest)
        except Exception as e:
            logger.warning(f"Failed to save ELA diagnostic visualization: {e}")
            return None

