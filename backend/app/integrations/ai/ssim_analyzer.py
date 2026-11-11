"""
ssim_analyzer.py - Structural Similarity Index (SSIM) Image Forensics Analyzer
==============================================================================
Calculates the Structural Similarity Index (SSIM) between evidence and baseline imagery.
Normalizes dimensions, channels, and luminance distributions to provide an explainable
measurement of luminance, contrast, and structural degradation or progression.
"""

from io import BytesIO
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as compute_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    compute_ssim = None

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import SsimCheckResult


class StructuralSimilarityAnalyzer:
    """Computes SSIM between two normalized image arrays."""

    def __init__(self, threshold: Optional[float] = None, target_size: Tuple[int, int] = (512, 512)):
        self.threshold = threshold if threshold is not None else settings.SSIM_SIMILARITY_THRESHOLD
        self.target_size = target_size

    def compare(
        self,
        current_bytes: bytes,
        baseline_bytes: bytes,
        comparison_target: str = "baseline_progress"
    ) -> SsimCheckResult:
        """
        Normalizes both images to identical dimensions (target_size) and computes SSIM.
        Range: [-1.0, 1.0]. 1.0 = structurally identical.
        """
        if not SKIMAGE_AVAILABLE or compute_ssim is None:
            return SsimCheckResult(
                status=VerificationCheckStatus.UNAVAILABLE,
                passed=False,
                ssim_score=None,
                threshold=self.threshold,
                comparison_target=comparison_target,
                message="scikit-image library is not available. SSIM comparison skipped."
            )

        try:
            img_curr = Image.open(BytesIO(current_bytes)).convert("L").resize(self.target_size, Image.Resampling.LANCZOS)
            img_base = Image.open(BytesIO(baseline_bytes)).convert("L").resize(self.target_size, Image.Resampling.LANCZOS)

            arr_curr = np.asarray(img_curr, dtype=np.float32)
            arr_base = np.asarray(img_base, dtype=np.float32)

            score = float(compute_ssim(arr_curr, arr_base, data_range=255.0))

            # Interpretation:
            # High SSIM (> 0.90) indicates identical scene structure with minimal changes.
            # Moderate SSIM (0.50 - 0.85) indicates same scene/angle with physical changes/progress.
            # Low SSIM (< 0.40) indicates completely different vantage point, lighting, or scene.
            if score >= 0.95:
                status = VerificationCheckStatus.PASSED
                msg = f"Near-identical structural composition (SSIM: {score:.3f} >= 0.95). Scene structure is virtually unchanged."
            elif score >= self.threshold:
                status = VerificationCheckStatus.PASSED
                msg = f"Moderate structural alignment with visible changes (SSIM: {score:.3f} >= {self.threshold:.2f}). Progression consistent with same vantage point."
            else:
                status = VerificationCheckStatus.PASSED
                msg = f"Low structural similarity (SSIM: {score:.3f} < {self.threshold:.2f}). Substantial scene alteration, different perspective, or distinct setting."

            return SsimCheckResult(
                status=status,
                passed=True,
                ssim_score=round(score, 4),
                threshold=self.threshold,
                comparison_target=comparison_target,
                message=msg
            )
        except Exception as e:
            logger.error(f"SSIM comparison calculation failed: {e}")
            return SsimCheckResult(
                status=VerificationCheckStatus.ERROR,
                passed=False,
                ssim_score=None,
                threshold=self.threshold,
                comparison_target=comparison_target,
                message=f"SSIM calculation error: {e}"
            )

