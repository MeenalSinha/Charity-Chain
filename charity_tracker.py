"""
charity_tracker.py - Core Backend & Domain Logic Module
======================================================
Business logic, database management, computer vision verification,
forensic indicators, and analytics.

CharityChain Core Principle:
    AI analyzes. Evidence supports. Humans decide. Payment systems execute. CharityChain records.
"""

import os
import sys
import json
import time
import hashlib
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import base64
import mimetypes
import asyncio
from math import radians, sin, cos, sqrt, atan2

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Core numeric & imaging libraries
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ExifTags
import imagehash
import requests
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

# Optional imports with availability guards
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from tinydb import TinyDB, Query
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central application configuration (Phase 0 baseline)"""
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    WEB3_PROVIDER = os.getenv('WEB3_PROVIDER', 'https://rpc-mumbai.maticvigil.com')
    ORACLE_PRIVATE_KEY = os.getenv('ORACLE_PRIVATE_KEY', 'DEMO_MODE')
    WEB3_STORAGE_TOKEN = os.getenv('WEB3_STORAGE_TOKEN', 'DEMO_MODE')
    PINATA_API_KEY = os.getenv('PINATA_API_KEY', 'DEMO_MODE')
    PINATA_SECRET = os.getenv('PINATA_SECRET', 'DEMO_MODE')
    CHAIN_ID = int(os.getenv('CHAIN_ID', '80001'))
    
    # Forensic & Verification Thresholds
    TAMPER_THRESHOLD = float(os.getenv('TAMPER_ELA_THRESHOLD', '12.0'))
    MIN_DETECTIONS = int(os.getenv('MIN_DETECTIONS', '1'))
    PHASH_DISTANCE_THRESHOLD = int(os.getenv('PHASH_DISTANCE_THRESHOLD', '10'))
    SSIM_THRESHOLD = float(os.getenv('SSIM_THRESHOLD', '0.4'))
    GPS_RADIUS_METERS = float(os.getenv('GPS_RADIUS_METERS', '1000'))
    CONFIDENCE_THRESHOLD = float(os.getenv('VERIFICATION_CONFIDENCE_THRESHOLD', '0.65'))
    
    # Rewards & Limits
    IMPACT_TOKEN_REWARD_RATE = 100
    MAX_UPLOAD_SIZE_MB = float(os.getenv('MAX_UPLOAD_SIZE_MB', '10.0'))
    IMAGE_RESIZE_WIDTH = 640
    CACHE_TTL_SECONDS = 300
    RATE_LIMIT_UPLOADS_PER_HOUR = int(os.getenv('RATE_LIMIT_UPLOADS_PER_HOUR', '10'))
    
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
    
    DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
    EVIDENCE_DIR = DATA_DIR / 'evidence'
    BUILD_DIR = Path('build')
    DB_FILE = DATA_DIR / 'charity.db'
    MODELS_DIR = Path('models')
    LOGS_DIR = Path('logs')
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
    
    @staticmethod
    def mask_sensitive(value: str, show_chars: int = 4) -> str:
        if not value or value in ('DEMO_MODE', 'optional'):
            return 'UNSET_OR_DEMO'
        return f"{value[:show_chars]}...{value[-show_chars:]}"


# Ensure baseline runtime directories exist
for directory in [Config.DATA_DIR, Config.EVIDENCE_DIR, Config.BUILD_DIR, 
                  Config.MODELS_DIR, Config.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    """Standardized structured logger for operational events"""
    @staticmethod
    def log(level: str, message: str, data: Optional[Dict] = None):
        try:
            log_file = Config.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'data': data or {}
            }
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            # Fallback to stderr if file logging fails
            sys.stderr.write(f"[{level}] {message}\n")


# ============================================================================
# SECURITY VALIDATOR
# ============================================================================

class SecurityValidator:
    """Validates uploads, formats, and basic access rate limits"""
    @staticmethod
    def validate_mime_type(file_bytes: bytes) -> Tuple[bool, str]:
        if len(file_bytes) < 8:
            return False, 'Unknown/Too Short'
        if file_bytes[:2] == b'\xff\xd8':
            mime_type = 'image/jpeg'
        elif file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = 'image/png'
        else:
            return False, 'Unknown'
        return mime_type in Config.ALLOWED_MIME_TYPES, mime_type
    
    @staticmethod
    def validate_file_size(file_bytes: bytes) -> Tuple[bool, float]:
        size_mb = len(file_bytes) / (1024 * 1024)
        return size_mb <= Config.MAX_UPLOAD_SIZE_MB, size_mb
    
    @staticmethod
    def check_rate_limit(user_id: str) -> bool:
        rate_file = Config.DATA_DIR / f'rate_{user_id}.json'
        if not rate_file.exists():
            return True
        try:
            with open(rate_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cutoff = datetime.now() - timedelta(hours=1)
            recent = [ts for ts in data.get('timestamps', []) if datetime.fromisoformat(ts) > cutoff]
            return len(recent) < Config.RATE_LIMIT_UPLOADS_PER_HOUR
        except Exception:
            return True
    
    @staticmethod
    def record_upload(user_id: str):
        try:
            rate_file = Config.DATA_DIR / f'rate_{user_id}.json'
            data = {'timestamps': []}
            if rate_file.exists():
                with open(rate_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            data['timestamps'].append(datetime.now().isoformat())
            cutoff = datetime.now() - timedelta(hours=1)
            data['timestamps'] = [ts for ts in data['timestamps'] if datetime.fromisoformat(ts) > cutoff]
            with open(rate_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            Logger.log('WARNING', f"Failed to record upload rate: {e}")


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """
    TinyDB Database Layer (Phase 0 Baseline).
    Fixes the TinyDB truthiness bug by strictly checking `self.db is not None`.
    Provides deterministic table initialization and non-destructive operations.
    """
    def __init__(self, db_file: Optional[Path] = None):
        self.db_file = db_file or Config.DB_FILE
        self.db = None
        self.milestones = None
        self.donations = None
        self.verifications = None
        self.ngo_profiles = None
        self.nfts = None
        self.analytics = None
        
        if DB_AVAILABLE:
            try:
                self.db_file.parent.mkdir(parents=True, exist_ok=True)
                self.db = TinyDB(str(self.db_file))
                self._init_tables()
                Logger.log('INFO', f'Database opened at {self.db_file}: {len(self.milestones.all())} milestones found')
            except Exception as e:
                Logger.log('ERROR', f'Failed to open database at {self.db_file}: {e}')
                self.db = None

    def _init_tables(self):
        """Deterministically initialize all table handles."""
        if self.db is not None:
            self.milestones = self.db.table('milestones')
            self.donations = self.db.table('donations')
            self.verifications = self.db.table('verifications')
            self.ngo_profiles = self.db.table('ngo_profiles')
            self.nfts = self.db.table('nfts')
            self.analytics = self.db.table('analytics')
    
    def close(self):
        """Safely close database connection"""
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    def create_milestone(self, owner: str, goal: int, description: str, 
                         latitude: float = 0.0, longitude: float = 0.0) -> int:
        """Create a new project milestone with collision-safe ID generation."""
        all_milestones = self.get_all_milestones()
        milestone_id = max([m.get('id', 0) for m in all_milestones], default=0) + 1
        milestone = {
            'id': milestone_id,
            'owner': owner,
            'goal': goal,
            'raised': 0,
            'status': 'PENDING',
            'description': description,
            'latitude': latitude,
            'longitude': longitude,
            'created_at': datetime.now().isoformat(),
            'evidence_cid': None,
            'verification_score': None
        }
        if self.db is not None and self.milestones is not None and self.ngo_profiles is not None:
            self.milestones.insert(milestone)
            Query_ = Query()
            profile = self.ngo_profiles.search(Query_.address == owner)
            if not profile:
                self.ngo_profiles.insert({
                    'address': owner,
                    'reputation_score': 100,
                    'completed_milestones': 0,
                    'rejected_milestones': 0,
                    'total_raised': 0
                })
        return milestone_id
    
    def add_donation(self, milestone_id: int, donor: str, amount: int):
        """Record a donation and update raised amount."""
        donation = {
            'milestone_id': milestone_id,
            'donor': donor,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'impact_tokens_earned': int(amount * Config.IMPACT_TOKEN_REWARD_RATE / 10**18)
        }
        if self.db is not None and self.donations is not None and self.milestones is not None:
            self.donations.insert(donation)
            Query_ = Query()
            milestone = self.milestones.get(Query_.id == milestone_id)
            if milestone:
                self.milestones.update({'raised': milestone['raised'] + amount}, Query_.id == milestone_id)
    
    def submit_evidence(self, milestone_id: int, cid: str, verification_result: Dict):
        """Record submitted evidence and verification outcomes."""
        if self.db is not None and self.milestones is not None and self.verifications is not None:
            Query_ = Query()
            self.milestones.update({
                'evidence_cid': cid,
                'status': 'SUBMITTED',
                'verification_score': verification_result.get('confidence', 0),
                'updated_at': datetime.now().isoformat()
            }, Query_.id == milestone_id)
            self.verifications.insert({
                'milestone_id': milestone_id,
                'cid': cid,
                'result': verification_result,
                'timestamp': datetime.now().isoformat()
            })
            self.log_verification_analytics(milestone_id, verification_result)
    
    def log_verification_analytics(self, milestone_id: int, result: Dict):
        """Log verification metrics for platform analytics."""
        if self.db is not None and self.analytics is not None:
            checks = result.get('checks', {})
            passed_checks = sum(1 for c in checks.values() if isinstance(c, dict) and c.get('passed'))
            self.analytics.insert({
                'milestone_id': milestone_id,
                'timestamp': datetime.now().isoformat(),
                'confidence': result.get('confidence', 0),
                'checks_passed': passed_checks,
                'total_checks': len(checks),
                'result': 'PASS' if result.get('pass') else 'FAIL'
            })
    
    def update_milestone_status(self, milestone_id: int, status: str):
        """Update milestone lifecycle status. On RELEASED, issues impact certificates."""
        if self.db is not None and self.milestones is not None:
            Query_ = Query()
            milestone = self.milestones.get(Query_.id == milestone_id)
            if milestone:
                self.milestones.update({'status': status}, Query_.id == milestone_id)
                if status == 'RELEASED':
                    self._mint_nfts_for_milestone(milestone_id)
    
    def _mint_nfts_for_milestone(self, milestone_id: int):
        """
        Issues verifiable Proof-of-Impact digital certificates for donors of a released milestone.
        Note: These are locally signed and verifiable digital records.
        """
        if self.db is not None and self.nfts is not None and self.donations is not None:
            Query_ = Query()
            donations = self.donations.search(Query_.milestone_id == milestone_id)
            for donation in donations:
                all_nfts = self.nfts.all() or []
                nft_id = max([n.get('nft_id', 0) for n in all_nfts], default=0) + 1
                cert_hash = hashlib.sha256(
                    f"{donation['donor']}:{milestone_id}:{donation['amount']}:{datetime.now().isoformat()}".encode()
                ).hexdigest()
                self.nfts.insert({
                    'nft_id': nft_id,
                    'owner': donation['donor'],
                    'milestone_id': milestone_id,
                    'donation_amount': donation['amount'],
                    'minted_at': datetime.now().isoformat(),
                    'token_uri': f"certificate://proof-of-impact/{cert_hash[:20]}",
                    'metadata': {
                        'name': f"Proof of Impact Certificate #{nft_id}",
                        'description': f"Verified charitable contribution record for milestone #{milestone_id}",
                        'record_type': 'LOCAL_DIGITAL_CERTIFICATE',
                        'certificate_hash': cert_hash
                    }
                })
    
    def get_all_milestones(self) -> List[Dict]:
        """Retrieve all registered milestones."""
        if DB_AVAILABLE and self.db is not None and self.milestones is not None:
            try:
                results = self.milestones.all()
                return results if results else []
            except Exception as e:
                Logger.log('ERROR', f'get_all_milestones failed: {e}')
                return []
        return []
    
    def get_milestone(self, milestone_id: int) -> Optional[Dict]:
        """Retrieve a specific milestone by integer ID."""
        if self.db is not None and self.milestones is not None:
            Query_ = Query()
            return self.milestones.get(Query_.id == milestone_id)
        return None
    
    def get_donations_for_milestone(self, milestone_id: int) -> List[Dict]:
        """Retrieve all donations associated with a milestone."""
        if self.db is not None and self.donations is not None:
            Query_ = Query()
            return self.donations.search(Query_.milestone_id == milestone_id)
        return []
    
    def get_ngo_profile(self, ngo_address: str) -> Dict:
        """Retrieve reputation profile for an NGO."""
        if self.db is not None and self.ngo_profiles is not None:
            Query_ = Query()
            profile = self.ngo_profiles.get(Query_.address == ngo_address)
            return profile if profile else {
                'reputation_score': 100,
                'completed_milestones': 0,
                'rejected_milestones': 0,
                'total_raised': 0
            }
        return {'reputation_score': 100, 'completed_milestones': 0, 'rejected_milestones': 0, 'total_raised': 0}
    
    def get_donor_nfts(self, donor_address: str) -> List[Dict]:
        """Retrieve Proof-of-Impact certificates owned by a donor."""
        if self.db is not None and self.nfts is not None:
            Query_ = Query()
            return self.nfts.search(Query_.owner == donor_address)
        return []
    
    def get_analytics_summary(self) -> Dict:
        """Retrieve aggregated platform verification statistics."""
        if self.db is None or self.analytics is None:
            return {}
        all_analytics = self.analytics.all()
        if not all_analytics:
            return {
                'total_verifications': 0,
                'avg_confidence': 0,
                'pass_rate': 0,
                'avg_gps_distance': 0,
                'avg_objects_detected': 0
            }
        return {
            'total_verifications': len(all_analytics),
            'avg_confidence': float(np.mean([a.get('confidence', 0) for a in all_analytics])),
            'pass_rate': len([a for a in all_analytics if a.get('result') == 'PASS']) / len(all_analytics) * 100
        }


# ============================================================================
# AI EVIDENCE VERIFIER & FORENSIC ENGINE
# ============================================================================

class EnhancedAIVerifier:
    """
    Evidence Verification Engine.
    Executes genuine computer vision and forensic indicator checks.
    Distinguishes PASSED, FAILED, UNAVAILABLE, NOT_APPLICABLE, and PENDING.
    Never returns fabricated detections or mock GPS distances.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.yolo_model = None
        self.yolo_model_path = model_path or Config.YOLO_MODEL_PATH
        if YOLO_AVAILABLE:
            try:
                if Path(self.yolo_model_path).exists():
                    self.yolo_model = YOLO(self.yolo_model_path)
                    Logger.log('INFO', f"YOLO model loaded from {self.yolo_model_path}")
                else:
                    Logger.log('WARNING', f"YOLO model weights file not found at {self.yolo_model_path}")
            except Exception as e:
                Logger.log('ERROR', f"YOLO initialization failed: {e}")
                self.yolo_model = None
    
    def detect_objects(self, image_bytes: bytes, conf_threshold: float = 0.3) -> Tuple[List[Dict], str]:
        """
        Runs genuine YOLO object detection.
        Returns: (detections_list, status)
        Status is 'PASSED', 'FAILED', or 'UNAVAILABLE'.
        Never returns fabricated detections.
        """
        if not self.yolo_model:
            return [], "UNAVAILABLE"
        
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix='.jpg')
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                f.write(image_bytes)
            
            results = self.yolo_model.predict(source=tmp_path_str, conf=conf_threshold, save=False, imgsz=640)
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': int(box.cls[0]),
                        'conf': float(box.conf[0]),
                        'xyxy': box.xyxy[0].tolist(),
                        'label': result.names[int(box.cls[0])]
                    })
            status = "PASSED" if len(detections) >= Config.MIN_DETECTIONS else "FAILED"
            return detections, status
        except Exception as e:
            Logger.log('ERROR', f"Object detection failed during inference: {e}")
            return [], "FAILED"
        finally:
            if os.path.exists(tmp_path_str):
                try:
                    os.unlink(tmp_path_str)
                except Exception:
                    pass
    
    def get_annotated_image(self, image_bytes: bytes, detections: List[Dict]) -> bytes:
        """Draws red bounding boxes and confidence labels on detected objects."""
        if not detections:
            return image_bytes
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            draw = ImageDraw.Draw(img)
            for det in detections:
                xyxy = det.get('xyxy', [0, 0, 0, 0])
                draw.rectangle(xyxy, outline='red', width=3)
                text = f"{det.get('label', 'obj')}: {det.get('conf', 0.0):.2f}"
                draw.text((xyxy[0], max(0, xyxy[1] - 15)), text, fill='red')
            output = BytesIO()
            img.save(output, format='JPEG')
            return output.getvalue()
        except Exception as e:
            Logger.log('WARNING', f"Failed to annotate image: {e}")
            return image_bytes
    
    def get_exif_data(self, image_bytes: bytes) -> Dict:
        """Extracts human-readable EXIF tags from image bytes."""
        try:
            img = Image.open(BytesIO(image_bytes))
            exif = img._getexif() or {}
            readable = {}
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                # Avoid large binary blobs in readable output
                readable[str(decoded)] = str(val)[:100]
            return readable
        except Exception:
            return {}

    @staticmethod
    def _convert_dms_to_degrees(value) -> Optional[float]:
        """Converts GPS coordinate tuple (degrees, minutes, seconds) to decimal degrees."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        except Exception:
            return None

    def extract_exif_gps(self, image_bytes: bytes) -> Optional[Tuple[float, float]]:
        """
        Extracts genuine GPS latitude and longitude from image EXIF if present.
        Returns: (latitude, longitude) in decimal degrees, or None if not found/unparseable.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            exif = img._getexif()
            if not exif:
                return None
            
            gps_info = {}
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == 'GPSInfo' or tag_id == 34853:
                    if isinstance(value, dict):
                        for gps_tag_id, gps_value in value.items():
                            gps_tag_name = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_info[gps_tag_name] = gps_value
            
            if not gps_info:
                return None
            
            lat_val = gps_info.get('GPSLatitude')
            lat_ref = gps_info.get('GPSLatitudeRef', 'N')
            lon_val = gps_info.get('GPSLongitude')
            lon_ref = gps_info.get('GPSLongitudeRef', 'E')
            
            if lat_val and lon_val:
                lat = self._convert_dms_to_degrees(lat_val)
                lon = self._convert_dms_to_degrees(lon_val)
                if lat is not None and lon is not None:
                    if str(lat_ref).upper() == 'S':
                        lat = -lat
                    if str(lon_ref).upper() == 'W':
                        lon = -lon
                    return (lat, lon)
            return None
        except Exception as e:
            Logger.log('DEBUG', f"EXIF GPS extraction failed: {e}")
            return None

    def calculate_gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes Great-Circle distance in meters between two geocoordinates using Haversine formula."""
        R = 6371000  # Earth radius in meters
        phi1, lambda1, phi2, lambda2 = map(radians, [lat1, lon1, lat2, lon2])
        dphi = phi2 - phi1
        dlambda = lambda2 - lambda1
        a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    def verify_gps_location(self, image_bytes: bytes, expected_lat: float, expected_lon: float) -> Tuple[bool, str, Optional[float], str]:
        """
        Verifies evidence location against expected coordinates using genuine image EXIF GPS.
        Returns: (passed, message, distance_meters, status)
        Status is 'PASSED', 'FAILED', or 'UNAVAILABLE'.
        Never uses hardcoded distances.
        """
        coords = self.extract_exif_gps(image_bytes)
        if coords is None:
            return False, "No GPS metadata found in image EXIF. Location cannot be verified from image headers.", None, "UNAVAILABLE"
        
        actual_lat, actual_lon = coords
        distance = self.calculate_gps_distance(actual_lat, actual_lon, expected_lat, expected_lon)
        
        if distance <= Config.GPS_RADIUS_METERS:
            return True, f"GPS verified ({distance:.1f}m from expected milestone coordinates).", distance, "PASSED"
        else:
            return False, f"GPS location mismatch ({distance:.1f}m exceeds {Config.GPS_RADIUS_METERS}m tolerance).", distance, "FAILED"
    
    def error_level_analysis(self, image_bytes: bytes) -> Tuple[float, str]:
        """
        Performs Error Level Analysis (ELA) as a compression variance indicator.
        NOTE: ELA highlights compression error variations and is an investigative indicator,
        not absolute forensic proof of digital manipulation.
        Returns: (ela_score, status) where status is 'PASSED' or 'FAILED'
        """
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            buffer = BytesIO()
            img.save(buffer, 'JPEG', quality=90)
            buffer.seek(0)
            recompressed = Image.open(buffer)
            ela = ImageChops.difference(img, recompressed)
            arr = np.asarray(ela).astype(np.float32)
            score = float(arr.mean())
            status = "PASSED" if score < Config.TAMPER_THRESHOLD else "FAILED"
            return score, status
        except Exception as e:
            Logger.log('ERROR', f"ELA computation failed: {e}")
            return 0.0, "FAILED"
    
    def perceptual_hash_similarity(self, img_bytes_a: bytes, img_bytes_b: bytes) -> Tuple[Optional[int], str]:
        """Computes Hamming distance between perceptual hashes (pHash) of two images."""
        try:
            img_a = Image.open(BytesIO(img_bytes_a)).convert('RGB')
            img_b = Image.open(BytesIO(img_bytes_b)).convert('RGB')
            hash_a = imagehash.phash(img_a)
            hash_b = imagehash.phash(img_b)
            distance = int(hash_a - hash_b)
            return distance, "PASSED"
        except Exception as e:
            Logger.log('ERROR', f"pHash comparison failed: {e}")
            return None, "FAILED"
    
    def run_parallel_verification(self, evidence_bytes: bytes, before_bytes: Optional[bytes] = None,
                                  expected_lat: Optional[float] = None, expected_lon: Optional[float] = None,
                                  use_mock: bool = False) -> Dict:
        """
        Orchestrates multi-indicator verification across all available forensic modules.
        Distinguishes PASSED, FAILED, UNAVAILABLE, and NOT_APPLICABLE.
        Only executed, applicable checks contribute to the normalized confidence score.
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'pass': False,
            'confidence': 0.0
        }
        
        # 1. Location / GPS Check
        if expected_lat is not None and expected_lon is not None and (expected_lat != 0.0 or expected_lon != 0.0):
            gps_ok, gps_msg, gps_dist, gps_status = self.verify_gps_location(evidence_bytes, expected_lat, expected_lon)
            results['checks']['gps'] = {
                'status': gps_status,
                'passed': gps_ok,
                'message': gps_msg,
                'distance_meters': round(gps_dist, 1) if gps_dist is not None else None,
                'expected_location': {'lat': expected_lat, 'lon': expected_lon}
            }
        else:
            gps_coords = self.extract_exif_gps(evidence_bytes)
            if gps_coords:
                results['checks']['gps'] = {
                    'status': 'PASSED',
                    'passed': True,
                    'message': f"GPS metadata present ({gps_coords[0]:.4f}, {gps_coords[1]:.4f}). No project coordinates specified for proximity comparison.",
                    'distance_meters': None,
                    'extracted_location': {'lat': gps_coords[0], 'lon': gps_coords[1]}
                }
            else:
                results['checks']['gps'] = {
                    'status': 'NOT_APPLICABLE',
                    'passed': False,
                    'message': "No expected project coordinates configured; no EXIF GPS required.",
                    'distance_meters': None
                }
        
        # 2. Tamper / ELA Check (Compression Variance Indicator)
        tamper_score, tamper_status = self.error_level_analysis(evidence_bytes)
        tamper_ok = (tamper_status == "PASSED")
        results['checks']['tamper'] = {
            'status': tamper_status,
            'passed': tamper_ok,
            'ela_score': round(tamper_score, 2),
            'threshold': Config.TAMPER_THRESHOLD,
            'message': (f"Compression variance ({tamper_score:.2f}) within normal range." if tamper_ok 
                        else f"High compression variance ({tamper_score:.2f} >= {Config.TAMPER_THRESHOLD}); possible editing indicator.")
        }
        
        # 3. Object Detection Check (YOLOv8)
        detections, det_status = self.detect_objects(evidence_bytes)
        det_ok = (det_status == "PASSED")
        if det_status == "UNAVAILABLE":
            det_msg = "YOLO object detection model unavailable. Check skipped."
        elif det_ok:
            det_msg = f"Detected {len(detections)} relevant object(s) in evidence."
        else:
            det_msg = f"Insufficient relevant objects detected ({len(detections)} < {Config.MIN_DETECTIONS})."
        
        results['checks']['object_detection'] = {
            'status': det_status,
            'passed': det_ok,
            'total_objects': len(detections),
            'top_detections': detections[:5],
            'message': det_msg
        }
        
        # 4. Before / After Comparison Check
        if before_bytes is not None:
            phash_dist, phash_status = self.perceptual_hash_similarity(before_bytes, evidence_bytes)
            if phash_dist is not None:
                comparison_ok = phash_dist > Config.PHASH_DISTANCE_THRESHOLD
                results['checks']['before_after'] = {
                    'status': 'PASSED' if comparison_ok else 'FAILED',
                    'passed': comparison_ok,
                    'phash_distance': phash_dist,
                    'threshold': Config.PHASH_DISTANCE_THRESHOLD,
                    'message': (f"Distinct visual progression detected (pHash distance {phash_dist} > {Config.PHASH_DISTANCE_THRESHOLD})." if comparison_ok
                                else f"Insufficient visual difference (pHash distance {phash_dist} <= {Config.PHASH_DISTANCE_THRESHOLD}); possible duplicate evidence.")
                }
            else:
                results['checks']['before_after'] = {
                    'status': 'FAILED',
                    'passed': False,
                    'phash_distance': None,
                    'message': 'Failed to compute perceptual hash comparison between before and after images.'
                }
        else:
            results['checks']['before_after'] = {
                'status': 'NOT_APPLICABLE',
                'passed': False,
                'phash_distance': None,
                'message': 'No baseline (before) photograph provided for comparative verification.'
            }
        
        # Normalized Confidence Calculation
        # Only actually executed checks with PASSED or FAILED status contribute to the denominator
        weights = {'gps': 0.25, 'tamper': 0.30, 'object_detection': 0.30, 'before_after': 0.15}
        active_weights = 0.0
        earned_weights = 0.0
        for check_name, weight in weights.items():
            check = results['checks'].get(check_name)
            if check and check.get('status') in ['PASSED', 'FAILED']:
                active_weights += weight
                if check.get('passed'):
                    earned_weights += weight
        
        if active_weights > 0.0:
            confidence = round(earned_weights / active_weights, 2)
        else:
            confidence = 0.0
        
        results['confidence'] = min(1.0, confidence)
        results['pass'] = (results['confidence'] >= Config.CONFIDENCE_THRESHOLD and any(c.get('passed') for c in results['checks'].values()))
        results['annotated_image'] = self.get_annotated_image(evidence_bytes, detections) if detections else None
        
        return results


# ============================================================================
# IPFS & STORAGE MANAGER
# ============================================================================

class EnhancedIPFSManager:
    """
    Storage & Evidence Addressing Manager.
    Accurately distinguishes local content hashes vs genuine external IPFS storage.
    """
    @staticmethod
    def upload_to_web3storage(file_bytes: bytes, filename: str) -> Tuple[str, str, str]:
        """
        Uploads evidence bytes or calculates local content hash.
        Returns: (cid_or_hash, access_url, storage_mode)
        Storage mode explicitly indicates: 'EXTERNAL_IPFS' or 'LOCAL_STORAGE'
        """
        token = Config.WEB3_STORAGE_TOKEN
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        
        if not token or token == 'DEMO_MODE':
            local_cid = f"local-sha256-{content_hash[:32]}"
            return local_cid, f"local-evidence://{local_cid}/{filename}", "LOCAL_STORAGE"
        
        try:
            url = "https://api.web3.storage/upload"
            headers = {"Authorization": f"Bearer {token}"}
            files = {"file": (filename, file_bytes)}
            resp = requests.post(url, files=files, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cid = data.get('cid', f"sha256-{content_hash[:32]}")
            return cid, f"https://{cid}.ipfs.dweb.link/{filename}", "EXTERNAL_IPFS"
        except Exception as e:
            Logger.log('WARNING', f"Remote IPFS upload failed: {e}; falling back to local content address.")
            local_cid = f"local-sha256-{content_hash[:32]}"
            return local_cid, f"local-evidence://{local_cid}/{filename}", "LOCAL_STORAGE"
    
    @staticmethod
    async def upload_async(file_bytes: bytes, filename: str) -> Tuple[str, str, str]:
        """Async upload handler for evidence assets."""
        return EnhancedIPFSManager.upload_to_web3storage(file_bytes, filename)
    
    @staticmethod
    def generate_qr_code(cid: str) -> Optional[bytes]:
        """Generates a QR code image representing evidence content address."""
        if not QRCODE_AVAILABLE:
            return None
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(f"evidence://{cid}")
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
        except Exception:
            return None


# ============================================================================
# ASYNC IO MANAGER
# ============================================================================

class AsyncIOManager:
    @staticmethod
    async def verify_parallel_async(verifier: EnhancedAIVerifier, evidence_bytes: bytes,
                                    before_bytes: Optional[bytes],
                                    expected_lat: Optional[float],
                                    expected_lon: Optional[float]) -> Dict:
        """Run verification in a background worker thread."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, verifier.run_parallel_verification, evidence_bytes, before_bytes,
            expected_lat, expected_lon, False
        )
        return result


# ============================================================================
# VISUALIZATION MANAGER
# ============================================================================

class VisualizationManager:
    """Generates charts, gauges, and diagrams for analytics."""
    @staticmethod
    def create_confidence_gauge(confidence: float) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "AI Verification Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 65], 'color': "lightgray"},
                    {'range': [65, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def create_confidence_explanation_panel(checks: Dict, confidence: float) -> go.Figure:
        modules = []
        contributions = []
        colors = []
        weights = {'gps': 0.25, 'tamper': 0.30, 'object_detection': 0.30, 'before_after': 0.15}
        for check_name, weight in weights.items():
            if check_name in checks:
                check = checks[check_name]
                modules.append(check_name.replace('_', ' ').title())
                status = check.get('status', 'PASSED' if check.get('passed') else 'FAILED')
                if status == 'PASSED':
                    contributions.append(weight * 100)
                    colors.append('green')
                elif status == 'FAILED':
                    contributions.append(weight * 100)
                    colors.append('red')
                elif status == 'UNAVAILABLE':
                    contributions.append(weight * 50)
                    colors.append('gray')
                else:
                    contributions.append(0)
                    colors.append('lightgray')
        fig = go.Figure(data=[go.Barpolar(r=contributions, theta=modules, marker=dict(color=colors))])
        fig.update_layout(title="Forensic Check Breakdown", height=400)
        return fig
    
    @staticmethod
    def create_donations_over_time(donations: List[Dict]) -> go.Figure:
        by_date = {}
        for don in donations:
            date = don.get('timestamp', '')[:10]
            if date:
                by_date[date] = by_date.get(date, 0) + don.get('amount', 0) / 10**18
        dates = sorted(by_date.keys())
        amounts = [by_date[d] for d in dates]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=amounts, mode='lines+markers', name='Daily Raised'))
        fig.update_layout(title='Donations Over Time', height=400)
        return fig
    
    @staticmethod
    def create_timeline_chart(milestones: List[Dict]) -> alt.Chart:
        timeline_data = []
        for m in milestones:
            created = datetime.fromisoformat(m.get('created_at', datetime.now().isoformat()))
            timeline_data.append({
                'Milestone': f"#{m['id']}",
                'Date': created.strftime('%Y-%m-%d'),
                'Status': m.get('status', 'PENDING')
            })
        chart = alt.Chart(alt.Data(values=timeline_data)).mark_circle(size=200).encode(
            x='Date:T', y='Milestone:N', color='Status:N', tooltip=['Milestone', 'Status']
        ).properties(height=400).interactive()
        return chart


# ============================================================================
# ROLE MANAGER
# ============================================================================

class RoleManager:
    """Role selection and view guard for Streamlit interface."""
    ROLES = ['NGO', 'Donor', 'Reviewer', 'Guest']
    
    @staticmethod
    def login_page():
        if not STREAMLIT_AVAILABLE:
            return
        if 'user_role' not in st.session_state:
            st.session_state.user_role = 'Guest'
            st.session_state.user_id = 'demo_user'
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔐 Active Role")
        current_role = st.session_state.user_role
        idx = RoleManager.ROLES.index(current_role) if current_role in RoleManager.ROLES else 0
        selected = st.sidebar.selectbox("Select Role", RoleManager.ROLES, index=idx)
        if selected != st.session_state.user_role:
            st.session_state.user_role = selected
            st.rerun()
    
    @staticmethod
    def require_role(allowed_roles: List[str]) -> bool:
        if not STREAMLIT_AVAILABLE:
            return True
        if 'user_role' not in st.session_state:
            st.session_state.user_role = 'Guest'
        if st.session_state.user_role not in allowed_roles:
            st.warning(f"Access restricted. Requires role: {', '.join(allowed_roles)}")
            return False
        return True


# ============================================================================
# AI ASSISTANT (Rule-Based Fallback / Provider-Ready)
# ============================================================================

class AIAssistant:
    """
    CharityChain Assistant.
    Clearly distinguishes rule-based status responses from live external LLM inference.
    Never fabricates LLM responses when credentials are unconfigured.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.available = bool(self.api_key and self.api_key not in ('DEMO_MODE', 'optional', ''))
    
    def query(self, question: str, context: Dict) -> str:
        if not self.available:
            q = question.lower()
            if 'verified' in q:
                return (f"[Rule-Based Status] Project verification confidence: "
                        f"{context.get('confidence', 0)}% (Status: {context.get('status', 'PENDING')}). "
                        f"(Note: Live LLM inference provider not configured).")
            elif 'status' in q:
                return (f"[Rule-Based Status] Milestone Status: {context.get('status', 'PENDING')}. "
                        f"(Note: Live LLM inference provider not configured).")
            else:
                return (f"[Rule-Based Status] CharityChain Platform Summary — Total Projects: "
                        f"{context.get('total_projects', 0)}. (Note: Live LLM inference provider not configured).")
        return f"[Live Assistant] Response to '{question[:50]}'"


# ============================================================================
# BADGE & GAMIFICATION SYSTEM
# ============================================================================

class BadgeSystem:
    BADGES = {
        'top_donor': {'name': '💎 Top Donor', 'threshold': 5, 'metric': 'donation_count'},
        'early_supporter': {'name': '🚀 Early Supporter', 'threshold': 1, 'metric': 'first_donation'},
        'impact_champion': {'name': '🏆 Impact Champion', 'threshold': 10, 'metric': 'total_eth'},
        'trusted_ngo': {'name': '⭐ Trusted NGO', 'threshold': 10, 'metric': 'completed_milestones'},
        'verified_100': {'name': '✅ 100% Verified', 'threshold': 5, 'metric': 'verified_count'}
    }
    
    @staticmethod
    def check_badges(user_id: str, db: DatabaseManager) -> List[str]:
        earned = []
        donations = [d for d in (db.donations.all() if (db.db is not None and db.donations is not None) else []) if d.get('donor') == user_id]
        nfts = db.get_donor_nfts(user_id)
        
        donation_count = len(donations)
        total_eth = sum(d.get('amount', 0) for d in donations) / 10**18 if donations else 0
        nft_count = len(nfts)
        
        if donation_count >= 5:
            earned.append('top_donor')
        if donation_count >= 1:
            earned.append('early_supporter')
        if total_eth >= 10:
            earned.append('impact_champion')
        if nft_count >= 5:
            earned.append('verified_100')
        
        milestones = db.get_all_milestones()
        user_milestones = [m for m in milestones if m.get('owner') == user_id]
        completed = len([m for m in user_milestones if m.get('status') == 'RELEASED'])
        if completed >= 10:
            earned.append('trusted_ngo')
        
        return earned
    
    @staticmethod
    def display_badges(badges: List[str]):
        if not STREAMLIT_AVAILABLE:
            return
        if not badges:
            st.info("No badges earned yet. Contribute to verified projects to earn impact achievements!")
            return
        st.subheader("🏅 Earned Impact Badges")
        cols = st.columns(min(len(badges), 4))
        for i, badge_id in enumerate(badges):
            badge_info = BadgeSystem.BADGES.get(badge_id, {})
            with cols[i % 4]:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 10px; color: white; margin-bottom: 1rem;'>
                    <h4>{badge_info.get('name', badge_id)}</h4>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def get_badge_progress(user_id: str, db: DatabaseManager) -> Dict[str, Dict]:
        donations = [d for d in (db.donations.all() if (db.db is not None and db.donations is not None) else []) if d.get('donor') == user_id]
        nfts = db.get_donor_nfts(user_id)
        milestones = db.get_all_milestones()
        user_milestones = [m for m in milestones if m.get('owner') == user_id]
        
        donation_count = len(donations)
        total_eth = sum(d.get('amount', 0) for d in donations) / 10**18 if donations else 0
        nft_count = len(nfts)
        completed = len([m for m in user_milestones if m.get('status') == 'RELEASED'])
        
        return {
            'top_donor': {
                'current': donation_count, 'target': 5,
                'percentage': min(100, (donation_count / 5) * 100),
                'earned': donation_count >= 5
            },
            'early_supporter': {
                'current': donation_count, 'target': 1,
                'percentage': 100 if donation_count >= 1 else 0,
                'earned': donation_count >= 1
            },
            'impact_champion': {
                'current': total_eth, 'target': 10,
                'percentage': min(100, (total_eth / 10) * 100),
                'earned': total_eth >= 10
            },
            'verified_100': {
                'current': nft_count, 'target': 5,
                'percentage': min(100, (nft_count / 5) * 100),
                'earned': nft_count >= 5
            },
            'trusted_ngo': {
                'current': completed, 'target': 10,
                'percentage': min(100, (completed / 10) * 100),
                'earned': completed >= 10
            }
        }


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def apply_custom_css():
    """Apply styling to Streamlit UI."""
    if not STREAMLIT_AVAILABLE:
        return
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.3rem;
    }
    .metric-card p {
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .metric-card small {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initializes Streamlit session state."""
    if not STREAMLIT_AVAILABLE:
        return
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    if 'verifier' not in st.session_state:
        st.session_state.verifier = EnhancedAIVerifier()
    if 'viz' not in st.session_state:
        st.session_state.viz = VisualizationManager()
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'Guest'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'demo_user'
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

def add_dark_mode_toggle():
    if not STREAMLIT_AVAILABLE:
        return
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    if dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #1e1e1e; color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)

def create_project_map(milestones: List[Dict]):
    """Creates a Folium map of registered projects."""
    if not FOLIUM_AVAILABLE or not milestones:
        return None
    valid_milestones = [m for m in milestones if m.get('latitude', 0) != 0 and m.get('longitude', 0) != 0]
    if not valid_milestones:
        center_lat, center_lon = 20, 0
        zoom = 2
    else:
        center_lat = sum(m['latitude'] for m in valid_milestones) / len(valid_milestones)
        center_lon = sum(m['longitude'] for m in valid_milestones) / len(valid_milestones)
        zoom = 3
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='OpenStreetMap')
    status_colors = {
        'PENDING': 'blue', 'SUBMITTED': 'orange', 'VERIFIED': 'green',
        'REJECTED': 'red', 'RELEASED': 'purple'
    }
    for milestone in valid_milestones:
        lat, lon = milestone['latitude'], milestone['longitude']
        color = status_colors.get(milestone.get('status', 'PENDING'), 'gray')
        popup_html = f"""
        <div style="width: 240px;">
            <h4>#{milestone['id']}: {milestone.get('description', '')[:40]}...</h4>
            <p><strong>NGO:</strong> {milestone.get('owner', '')}</p>
            <p><strong>Status:</strong> {milestone.get('status', '')}</p>
            <p><strong>Goal:</strong> {milestone.get('goal', 0)/10**18:.2f} ETH</p>
            <p><strong>Raised:</strong> {milestone.get('raised', 0)/10**18:.4f} ETH</p>
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"#{milestone['id']}: {milestone.get('description', '')[:30]}...",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    return m

def show_nft_gallery(user_id: str, db: DatabaseManager):
    """Displays verifiable Proof-of-Impact certificates for a donor."""
    if not STREAMLIT_AVAILABLE:
        return
    nfts = db.get_donor_nfts(user_id)
    if not nfts:
        st.info("📜 No certificates yet. Donate to verified projects to earn Proof-of-Impact certificates!")
        st.markdown("""
        ### How Proof-of-Impact Works:
        1. **Donate** to any project milestone.
        2. NGO **submits photo evidence** of completed work.
        3. **AI & Human Reviewers** verify real-world impact.
        4. **Certificate generated** as proof of verified contribution!
        """)
        return
    
    st.success(f"🎉 You have **{len(nfts)}** Proof-of-Impact Certificate(s)!")
    badges = BadgeSystem.check_badges(user_id, db)
    if badges:
        st.markdown("---")
        BadgeSystem.display_badges(badges)
    
    st.markdown("---")
    st.subheader("🖼️ Proof-of-Impact Certificates")
    cols = st.columns(3)
    for i, nft in enumerate(nfts):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📜 Certificate #{nft.get('nft_id')}</h3>
                <p><strong>Milestone:</strong> #{nft.get('milestone_id')}</p>
                <p><strong>Amount:</strong> {nft.get('donation_amount', 0)/10**18:.4f} ETH</p>
                <p><strong>Issued:</strong> {nft.get('minted_at', '')[:10]}</p>
            </div>
            """, unsafe_allow_html=True)
            milestone = db.get_milestone(nft.get('milestone_id'))
            if milestone and milestone.get('evidence_cid'):
                st.caption(f"Ref: `{milestone['evidence_cid'][:25]}...`")

def show_analytics_dashboard():
    """Displays platform statistics and verification analytics."""
    if not STREAMLIT_AVAILABLE:
        return
    st.header("📊 Platform Impact Analytics")
    db = st.session_state.db
    milestones = db.get_all_milestones()
    donations = db.donations.all() if (db.db is not None and db.donations is not None) else []
    verifications = db.verifications.all() if (db.db is not None and db.verifications is not None) else []
    nfts = db.nfts.all() if (db.db is not None and db.nfts is not None) else []
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", len(milestones))
    with col2:
        total_raised = sum(d.get('amount', 0) for d in donations) / 10**18
        st.metric("Total Raised", f"{total_raised:.2f} ETH")
    with col3:
        verified = len([m for m in milestones if m.get('status') in ['VERIFIED', 'RELEASED']])
        st.metric("Verified Milestones", verified)
    with col4:
        st.metric("Certificates Issued", len(nfts))
    
    st.markdown("---")
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("📈 Donations Over Time")
        if donations:
            fig = VisualizationManager.create_donations_over_time(donations)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No donation data yet")
    
    with col_viz2:
        st.subheader("🎯 Project Status Distribution")
        if milestones:
            status_counts = {}
            for m in milestones:
                s = m.get('status', 'PENDING')
                status_counts[s] = status_counts.get(s, 0) + 1
            fig = go.Figure(data=[go.Pie(labels=list(status_counts.keys()), values=list(status_counts.values()), hole=0.4)])
            fig.update_layout(title="Status Breakdown", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No project data yet")
    
    st.markdown("---")
    st.subheader("🤖 AI Verification Statistics")
    if verifications:
        confidences = [v.get('result', {}).get('confidence', 0) * 100 for v in verifications]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with c2:
            passed = len([c for c in confidences if c >= Config.CONFIDENCE_THRESHOLD * 100])
            st.metric("Passed Verifications", passed)
        with c3:
            failed = len([c for c in confidences if c < Config.CONFIDENCE_THRESHOLD * 100])
            st.metric("Failed/Flagged Verifications", failed)
    else:
        st.info("No verification records logged yet.")

def generate_demo_data(force: bool = False) -> bool:
    """
    Safely seeds initial demo projects and donations without crashing or silently corrupting data.
    """
    if not STREAMLIT_AVAILABLE or "db" not in st.session_state:
        Logger.log("ERROR", "Cannot generate demo data: Streamlit or session db not available")
        return False
    
    db: DatabaseManager = st.session_state.db
    
    if force and db.db is not None:
        db.db.drop_tables()
        db._init_tables()
    
    current = db.get_all_milestones()
    if current and not force:
        Logger.log("INFO", "Projects already exist, skipping demo data load", {'count': len(current)})
        return True
    
    demo = [
        ("Green Earth Foundation", "Plant 1000 trees in Amazon rainforest to combat deforestation", -3.4653, -62.2159, 2.5),
        ("Clean Water Initiative", "Build 10 wells in rural Kenya for 5000 people", -1.2921, 36.8219, 5.0),
        ("Education for All", "Provide books for 500 underprivileged children in Delhi", 28.6139, 77.2090, 3.0),
        ("Ocean Cleanup Project", "Remove 2 tons of plastic waste from Pacific coastal areas", 21.3099, -157.8581, 4.0),
        ("Medical Aid Africa", "Provide medical supplies for 3 rural clinics in Tanzania", -6.7924, 39.2083, 6.0),
    ]
    
    for i, (ngo, desc, lat, lon, eth) in enumerate(demo):
        milestone_id = db.create_milestone(
            owner=ngo,
            goal=int(eth * 10**18),
            description=desc,
            latitude=lat,
            longitude=lon
        )
        if i == 0:
            db.add_donation(milestone_id, "Alice Donor", int(0.5 * 10**18))
            db.add_donation(milestone_id, "Bob Supporter", int(0.3 * 10**18))
            db.add_donation(milestone_id, "Carol Benefactor", int(0.2 * 10**18))
        elif i == 1:
            db.add_donation(milestone_id, "David Philanthropist", int(2.0 * 10**18))
            db.add_donation(milestone_id, "Eva Contributor", int(1.5 * 10**18))
        elif i == 2:
            db.add_donation(milestone_id, "Frank Supporter", int(0.8 * 10**18))
    
    Logger.log("INFO", "Demo data generated successfully", {'projects': len(demo)})
    return True
