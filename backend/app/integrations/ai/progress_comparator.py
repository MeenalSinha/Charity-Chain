"""
progress_comparator.py - Multi-Indicator Baseline Progress Comparator
======================================================================
Compares newly submitted milestone evidence against an authorized baseline image.
Combines perceptual hash distance, structural similarity (SSIM), and generates
a side-by-side diagnostic visualization for human review transparency.
"""

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import ProgressCheckResult
from backend.app.integrations.ai.phash_analyzer import PerceptualHashAnalyzer
from backend.app.integrations.ai.ssim_analyzer import StructuralSimilarityAnalyzer


class ProgressComparator:
    """Evaluates progression between baseline evidence and newly submitted milestone evidence."""

    def __init__(self):
        self.phash_analyzer = PerceptualHashAnalyzer()
        self.ssim_analyzer = StructuralSimilarityAnalyzer()

    def compare_progression(
        self,
        current_bytes: bytes,
        baseline_bytes: Optional[bytes] = None,
        baseline_evidence_id: Optional[str] = None,
        baseline_hash: Optional[str] = None,
        generate_artifact: bool = True
    ) -> ProgressCheckResult:
        """
        Executes multi-indicator comparison against baseline evidence.
        """
        if baseline_bytes is None:
            return ProgressCheckResult(
                status=VerificationCheckStatus.NOT_APPLICABLE,
                passed=True,
                baseline_evidence_id=None,
                baseline_hash=None,
                phash_distance=None,
                ssim_score=None,
                progression_detected=False,
                message="No baseline evidence photograph was linked to this verification request. Progression comparison recorded as NOT_APPLICABLE.",
                diagnostic_artifact=None
            )

        try:
            phash_res = self.phash_analyzer.compare_images(current_bytes, baseline_bytes)
            ssim_res = self.ssim_analyzer.compare(current_bytes, baseline_bytes)

            phash_dist = phash_res.hamming_distance
            ssim_score = ssim_res.ssim_score

            # Progression determination:
            # - If pHash distance is 0 and SSIM > 0.98: exact duplicate file, no progression.
            # - If pHash distance is between 2 and 25 with SSIM between 0.4 and 0.85: consistent progression.
            # - If pHash distance > 25 and SSIM < 0.3: completely different subject or angle.
            is_duplicate = (phash_dist == 0) or (ssim_score is not None and ssim_score >= 0.98)
            progression_detected = not is_duplicate and (phash_dist is not None and phash_dist >= 2)

            if is_duplicate:
                status = VerificationCheckStatus.FLAGGED
                msg = f"Zero or negligible visual change detected (pHash distance {phash_dist}, SSIM: {ssim_score}). Possible duplicate reuse of identical baseline image without progression."
            elif progression_detected:
                status = VerificationCheckStatus.PASSED
                msg = f"Visible progression confirmed (pHash distance: {phash_dist}, SSIM: {ssim_score}). Consistent physical evolution relative to baseline."
            else:
                status = VerificationCheckStatus.PASSED
                msg = f"Progression indicators evaluated (pHash distance: {phash_dist}, SSIM: {ssim_score})."

            artifact_path = None
            if generate_artifact:
                artifact_path = self._generate_side_by_side_artifact(baseline_bytes, current_bytes, phash_dist, ssim_score)

            return ProgressCheckResult(
                status=status,
                passed=(status == VerificationCheckStatus.PASSED),
                baseline_evidence_id=baseline_evidence_id,
                baseline_hash=baseline_hash,
                phash_distance=phash_dist,
                ssim_score=ssim_score,
                progression_detected=progression_detected,
                message=msg,
                diagnostic_artifact=artifact_path
            )

        except Exception as e:
            logger.error(f"Progress comparison error: {e}")
            return ProgressCheckResult(
                status=VerificationCheckStatus.ERROR,
                passed=False,
                baseline_evidence_id=baseline_evidence_id,
                baseline_hash=baseline_hash,
                phash_distance=None,
                ssim_score=None,
                progression_detected=False,
                message=f"Progress comparison failed: {e}",
                diagnostic_artifact=None
            )

    def _generate_side_by_side_artifact(
        self,
        baseline_bytes: bytes,
        current_bytes: bytes,
        phash_dist: Optional[int],
        ssim_score: Optional[float]
    ) -> Optional[str]:
        """Creates a side-by-side composite diagnostic image (Baseline vs Current)."""
        try:
            art_dir = Path(settings.DIAGNOSTIC_ARTIFACTS_DIR)
            art_dir.mkdir(parents=True, exist_ok=True)

            img_base = Image.open(BytesIO(baseline_bytes)).convert("RGB").resize((400, 300))
            img_curr = Image.open(BytesIO(current_bytes)).convert("RGB").resize((400, 300))

            composite = Image.new("RGB", (820, 350), color=(240, 240, 240))
            composite.paste(img_base, (10, 40))
            composite.paste(img_curr, (410, 40))

            draw = ImageDraw.Draw(composite)
            draw.text((10, 15), "BASELINE EVIDENCE", fill=(50, 50, 50))
            draw.text((410, 15), "CURRENT EVIDENCE", fill=(50, 50, 50))

            stat_line = f"pHash Distance: {phash_dist if phash_dist is not None else 'N/A'} | SSIM: {ssim_score:.3f if ssim_score is not None else 'N/A'}"
            draw.text((10, 345), stat_line, fill=(80, 80, 80))

            filename = f"progression_compare_{int(time.time() * 1000)}_{os.getpid()}.jpg"
            dest = art_dir / filename
            composite.save(dest, format="JPEG", quality=85)
            return str(dest)
        except Exception as e:
            logger.warning(f"Failed to generate side-by-side comparison artifact: {e}")
            return None

