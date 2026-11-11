"""
phash_analyzer.py - Perceptual Hashing (pHash) Visual Similarity Analyzer
========================================================================
Calculates discrete cosine transform (DCT) perceptual hashes and Hamming distance
to evaluate visual similarity, detect duplicate evidence reuse, and support progression checks.
"""

from io import BytesIO
from typing import Optional, Tuple
from PIL import Image
import imagehash

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import PhashCheckResult


class PerceptualHashAnalyzer:
    """Computes and compares perceptual hashes to detect image similarity."""

    def __init__(self, distance_threshold: Optional[int] = None):
        self.distance_threshold = distance_threshold if distance_threshold is not None else settings.PHASH_DISTANCE_THRESHOLD

    def compute_phash(self, image_bytes: bytes) -> Optional[str]:
        """Calculates 64-bit hexadecimal pHash string for an image."""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            h = imagehash.phash(img)
            return str(h)
        except Exception as e:
            logger.error(f"pHash calculation failed: {e}")
            return None

    def compare_images(
        self,
        current_bytes: bytes,
        baseline_bytes: bytes
    ) -> PhashCheckResult:
        """
        Computes Hamming distance between perceptual hashes of two images.
        Distance 0: Perceptually identical.
        Distance 1-10: Highly similar, potential crop/resize/minor variation.
        Distance > 10: Substantially distinct imagery.
        """
        try:
            img_curr = Image.open(BytesIO(current_bytes)).convert("RGB")
            img_base = Image.open(BytesIO(baseline_bytes)).convert("RGB")

            hash_curr = imagehash.phash(img_curr)
            hash_base = imagehash.phash(img_base)

            hamming_distance = int(hash_curr - hash_base)

            # Interpretation logic
            if hamming_distance == 0:
                interpretation = "IDENTICAL_IMAGE"
                msg = f"Perceptually identical image detected (Hamming distance 0). Likely duplicate or exact reuse."
            elif hamming_distance <= 5:
                interpretation = "HIGHLY_SIMILAR"
                msg = f"High perceptual similarity (Hamming distance {hamming_distance} <= 5). Minor compression, re-saving, or subtle alteration."
            elif hamming_distance <= self.distance_threshold:
                interpretation = "MODERATELY_SIMILAR"
                msg = f"Moderate perceptual similarity (Hamming distance {hamming_distance} <= {self.distance_threshold})."
            else:
                interpretation = "DISTINCT"
                msg = f"Visually distinct images (Hamming distance {hamming_distance} > {self.distance_threshold}). Visual progression evident."

            return PhashCheckResult(
                status=VerificationCheckStatus.PASSED,
                passed=True,
                hamming_distance=hamming_distance,
                threshold=self.distance_threshold,
                similarity_interpretation=interpretation,
                message=msg
            )
        except Exception as e:
            logger.error(f"pHash comparison failed: {e}")
            return PhashCheckResult(
                status=VerificationCheckStatus.ERROR,
                passed=False,
                hamming_distance=None,
                threshold=self.distance_threshold,
                similarity_interpretation="ERROR",
                message=f"Failed to compute perceptual hash comparison: {e}"
            )

