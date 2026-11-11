"""
gps_analyzer.py - Geospatial Consistency & GPS Extraction Analyzer
===================================================================
Extracts authentic GPS geocoordinates from EXIF headers, validates coordinate bounds,
computes Great-Circle geodesic distance via the Haversine formula, and evaluates
proximity against expected project or milestone coordinates.
"""

from io import BytesIO
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple
from PIL import Image, ExifTags

from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import GpsCheckResult


class GpsGeospatialAnalyzer:
    """Extracts GPS coordinates from image metadata and evaluates distance to project site."""

    def __init__(self, tolerance_meters: Optional[float] = None):
        self.tolerance_meters = tolerance_meters if tolerance_meters is not None else settings.GPS_RADIUS_METERS

    @staticmethod
    def _convert_dms_to_degrees(dms_val) -> Optional[float]:
        """Converts degrees/minutes/seconds tuple or IFDRational into decimal degrees."""
        try:
            if isinstance(dms_val, (int, float)):
                return float(dms_val)
            d = float(dms_val[0])
            m = float(dms_val[1])
            s = float(dms_val[2])
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None

    def extract_coordinates(self, image_bytes: bytes) -> Optional[Tuple[float, float]]:
        """
        Extracts (latitude, longitude) from EXIF GPS tags.
        Returns None if not present or malformed.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            raw_exif = img._getexif()
            if not raw_exif:
                return None

            gps_info = {}
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name == "GPSInfo" or tag_id == 34853:
                    if isinstance(val, dict):
                        for sub_id, sub_val in val.items():
                            sub_name = ExifTags.GPSTAGS.get(sub_id, str(sub_id))
                            gps_info[sub_name] = sub_val

            if not gps_info:
                return None

            lat_raw = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef", "N")
            lon_raw = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef", "E")

            if not lat_raw or not lon_raw:
                return None

            lat = self._convert_dms_to_degrees(lat_raw)
            lon = self._convert_dms_to_degrees(lon_raw)

            if lat is None or lon is None:
                return None

            if str(lat_ref).strip().upper() == "S":
                lat = -lat
            if str(lon_ref).strip().upper() == "W":
                lon = -lon

            # Validate coordinate boundaries
            if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                return None

            return round(lat, 6), round(lon, 6)
        except Exception as e:
            logger.debug(f"GPS coordinate extraction failed: {e}")
            return None

    @staticmethod
    def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates Great-Circle geodesic distance in meters using Haversine formula."""
        R = 6371000.0  # Earth radius in meters
        phi1, lambda1, phi2, lambda2 = map(radians, [lat1, lon1, lat2, lon2])
        dphi = phi2 - phi1
        dlambda = lambda2 - lambda1
        a = sin(dphi / 2.0)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0)**2
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        return R * c

    def verify_location(
        self,
        image_bytes: bytes,
        expected_lat: Optional[float] = None,
        expected_lon: Optional[float] = None
    ) -> GpsCheckResult:
        """
        Evaluates geospatial consistency between extracted image EXIF coordinates
        and configured project/milestone coordinates.
        """
        extracted = self.extract_coordinates(image_bytes)

        if expected_lat is None or expected_lon is None:
            if extracted:
                return GpsCheckResult(
                    status=VerificationCheckStatus.PASSED,
                    passed=True,
                    extracted_latitude=extracted[0],
                    extracted_longitude=extracted[1],
                    expected_latitude=None,
                    expected_longitude=None,
                    distance_meters=None,
                    tolerance_meters=self.tolerance_meters,
                    message=f"Authentic GPS tags present in image ({extracted[0]:.4f}, {extracted[1]:.4f}). No expected project location coordinates configured for proximity comparison."
                )
            else:
                return GpsCheckResult(
                    status=VerificationCheckStatus.NOT_APPLICABLE,
                    passed=True,
                    extracted_latitude=None,
                    extracted_longitude=None,
                    expected_latitude=None,
                    expected_longitude=None,
                    distance_meters=None,
                    tolerance_meters=self.tolerance_meters,
                    message="No expected project location configured; evidence image lacks EXIF GPS tags. Geospatial check recorded as NOT_APPLICABLE."
                )

        if not extracted:
            return GpsCheckResult(
                status=VerificationCheckStatus.UNAVAILABLE,
                passed=False,
                extracted_latitude=None,
                extracted_longitude=None,
                expected_latitude=expected_lat,
                expected_longitude=expected_lon,
                distance_meters=None,
                tolerance_meters=self.tolerance_meters,
                message="No GPS metadata found in image EXIF. Geospatial proximity cannot be verified from image headers."
            )

        actual_lat, actual_lon = extracted
        distance = self.calculate_haversine_distance(actual_lat, actual_lon, expected_lat, expected_lon)
        within_tolerance = distance <= self.tolerance_meters

        status = VerificationCheckStatus.PASSED if within_tolerance else VerificationCheckStatus.FLAGGED
        if within_tolerance:
            msg = f"Geospatial consistency verified: Extracted coordinates are {distance:.1f} meters from expected location (within {self.tolerance_meters:.0f}m tolerance)."
        else:
            msg = f"Geospatial discrepancy: Extracted coordinates are {distance:.1f} meters from expected location, exceeding {self.tolerance_meters:.0f}m tolerance."

        return GpsCheckResult(
            status=status,
            passed=within_tolerance,
            extracted_latitude=actual_lat,
            extracted_longitude=actual_lon,
            expected_latitude=expected_lat,
            expected_longitude=expected_lon,
            distance_meters=round(distance, 1),
            tolerance_meters=self.tolerance_meters,
            message=msg
        )
