"""
exif_analyzer.py - Image EXIF Header & Metadata Extraction Analyzer
===================================================================
Extracts authentic Exchangeable Image File (EXIF) metadata from image bytes.
Distinguishes present, missing, and malformed metadata without ever fabricating fields.
"""

from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image, ExifTags

from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import ExifCheckResult


class ExifMetadataAnalyzer:
    """Extracts and sanitizes available EXIF metadata from evidence images."""

    def extract_metadata(self, image_bytes: bytes) -> ExifCheckResult:
        """
        Parses EXIF headers from image bytes.
        Returns ExifCheckResult with extracted timestamps, camera info, and dimensions.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            raw_exif = img._getexif()

            width, height = img.width, img.height

            if not raw_exif:
                return ExifCheckResult(
                    status=VerificationCheckStatus.UNAVAILABLE,
                    passed=True,
                    has_exif=False,
                    camera_make=None,
                    camera_model=None,
                    datetime_original=None,
                    image_width=width,
                    image_height=height,
                    has_gps=False,
                    message="No EXIF metadata headers found in evidence image. (Common for screenshots, web-saved, or privacy-scrubbed photos)."
                )

            # Map tag IDs to human-readable tag names
            decoded: Dict[str, Any] = {}
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(val, (bytes, bytearray)):
                    continue  # Skip large raw binary blobs
                decoded[tag_name] = val

            make = str(decoded.get("Make", "")).strip() or None
            model = str(decoded.get("Model", "")).strip() or None
            datetime_orig = str(decoded.get("DateTimeOriginal") or decoded.get("DateTime") or "").strip() or None
            has_gps = ("GPSInfo" in decoded or 34853 in raw_exif)

            details_msg = []
            if make or model:
                details_msg.append(f"Device: {make or ''} {model or ''}".strip())
            if datetime_orig:
                details_msg.append(f"Captured: {datetime_orig}")
            if has_gps:
                details_msg.append("GPS tags present")

            summary = "; ".join(details_msg) if details_msg else "EXIF present but key capture tags omitted"

            return ExifCheckResult(
                status=VerificationCheckStatus.PASSED,
                passed=True,
                has_exif=True,
                camera_make=make,
                camera_model=model,
                datetime_original=datetime_orig,
                image_width=width,
                image_height=height,
                has_gps=has_gps,
                message=f"Authentic EXIF headers parsed: {summary}."
            )
        except Exception as e:
            logger.debug(f"EXIF parsing failed: {e}")
            return ExifCheckResult(
                status=VerificationCheckStatus.UNAVAILABLE,
                passed=True,
                has_exif=False,
                camera_make=None,
                camera_model=None,
                datetime_original=None,
                image_width=None,
                image_height=None,
                has_gps=False,
                message=f"Could not parse EXIF metadata: {e}"
            )

