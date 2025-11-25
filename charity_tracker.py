"""
charity_tracker.py - Complete Backend Module
============================================
All logic, no UI. Import this from app.py

Usage:
    from charity_tracker import *
"""

import os
import sys
import json
import time
import hashlib
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import base64
import mimetypes
import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Core libraries
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ExifTags
import imagehash
import requests
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

# Optional imports
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
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

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
    """Application configuration"""
    WEB3_PROVIDER = os.getenv('WEB3_PROVIDER', 'https://rpc-mumbai.maticvigil.com')
    ORACLE_PRIVATE_KEY = os.getenv('ORACLE_PRIVATE_KEY', 'DEMO_MODE')
    WEB3_STORAGE_TOKEN = os.getenv('WEB3_STORAGE_TOKEN', 'DEMO_MODE')
    PINATA_API_KEY = os.getenv('PINATA_API_KEY', 'DEMO_MODE')
    PINATA_SECRET = os.getenv('PINATA_SECRET', 'DEMO_MODE')
    CHAIN_ID = int(os.getenv('CHAIN_ID', '80001'))
    
    TAMPER_THRESHOLD = 12.0
    MIN_DETECTIONS = 1
    PHASH_DISTANCE_THRESHOLD = 10
    SSIM_THRESHOLD = 0.4
    GPS_RADIUS_METERS = 1000
    CONFIDENCE_THRESHOLD = 0.65
    
    IMPACT_TOKEN_REWARD_RATE = 100
    MAX_UPLOAD_SIZE_MB = 10
    IMAGE_RESIZE_WIDTH = 640
    CACHE_TTL_SECONDS = 300
    RATE_LIMIT_UPLOADS_PER_HOUR = 10
    
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
    
    DATA_DIR = Path('data')
    EVIDENCE_DIR = DATA_DIR / 'evidence'
    BUILD_DIR = Path('build')
    DB_FILE = DATA_DIR / 'charity.db'
    MODELS_DIR = Path('models')
    LOGS_DIR = Path('logs')
    
    @staticmethod
    def mask_sensitive(value: str, show_chars: int = 4) -> str:
        if not value or value == 'DEMO_MODE':
            return 'DEMO_MODE'
        return f"{value[:show_chars]}...{value[-show_chars:]}"

# Create directories
for directory in [Config.DATA_DIR, Config.EVIDENCE_DIR, Config.BUILD_DIR, 
                  Config.MODELS_DIR, Config.LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    @staticmethod
    def log(level: str, message: str, data: Dict = None):
        log_file = Config.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'data': data or {}
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

# ============================================================================
# SECURITY VALIDATOR
# ============================================================================

class SecurityValidator:
    @staticmethod
    def validate_mime_type(file_bytes: bytes) -> Tuple[bool, str]:
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
        with open(rate_file, 'r') as f:
            data = json.load(f)
        cutoff = datetime.now() - timedelta(hours=1)
        recent = [ts for ts in data['timestamps'] if datetime.fromisoformat(ts) > cutoff]
        return len(recent) < Config.RATE_LIMIT_UPLOADS_PER_HOUR
    
    @staticmethod
    def record_upload(user_id: str):
        rate_file = Config.DATA_DIR / f'rate_{user_id}.json'
        data = {'timestamps': []}
        if rate_file.exists():
            with open(rate_file, 'r') as f:
                data = json.load(f)
        data['timestamps'].append(datetime.now().isoformat())
        cutoff = datetime.now() - timedelta(hours=1)
        data['timestamps'] = [ts for ts in data['timestamps'] if datetime.fromisoformat(ts) > cutoff]
        with open(rate_file, 'w') as f:
            json.dump(data, f)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self):
        # Always initialize these to None first
        self.db = None
        self.milestones = None
        self.donations = None
        self.verifications = None
        self.ngo_profiles = None
        self.nfts = None
        self.analytics = None
        
        # Now try to open database
        if DB_AVAILABLE:
            try:
                self.db = TinyDB(str(Config.DB_FILE))
                
                # Set table references
                self.milestones = self.db.table('milestones')
                self.donations = self.db.table('donations')
                self.verifications = self.db.table('verifications')
                self.ngo_profiles = self.db.table('ngo_profiles')
                self.nfts = self.db.table('nfts')
                self.analytics = self.db.table('analytics')
                
                Logger.log('INFO', f'Database opened: {len(self.milestones.all())} milestones found')
            except Exception as e:
                Logger.log('ERROR', f'Failed to open database: {e}')
                self.db = None
    
    # Remove the _init_tables method entirely - we don't need it
    
    def create_milestone(self, owner: str, goal: int, description: str, 
                        latitude: float = 0.0, longitude: float = 0.0) -> int:
        milestone_id = len(self.get_all_milestones()) + 1
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
        if self.db:
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
        donation = {
            'milestone_id': milestone_id,
            'donor': donor,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'impact_tokens_earned': int(amount * Config.IMPACT_TOKEN_REWARD_RATE / 10**18)
        }
        if self.db:
            self.donations.insert(donation)
            Query_ = Query()
            milestone = self.milestones.get(Query_.id == milestone_id)
            if milestone:
                self.milestones.update({'raised': milestone['raised'] + amount}, Query_.id == milestone_id)
    
    def submit_evidence(self, milestone_id: int, cid: str, verification_result: Dict):
        if self.db:
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
        if self.db:
            self.analytics.insert({
                'milestone_id': milestone_id,
                'timestamp': datetime.now().isoformat(),
                'confidence': result.get('confidence', 0),
                'checks_passed': sum(1 for c in result['checks'].values() if c['passed']),
                'total_checks': len(result['checks']),
                'result': 'PASS' if result['pass'] else 'FAIL'
            })
    
    def update_milestone_status(self, milestone_id: int, status: str):
        if self.db:
            Query_ = Query()
            milestone = self.milestones.get(Query_.id == milestone_id)
            if milestone:
                self.milestones.update({'status': status}, Query_.id == milestone_id)
                if status == 'RELEASED':
                    self._mint_nfts_for_milestone(milestone_id)
    
    def _mint_nfts_for_milestone(self, milestone_id: int):
        if self.db:
            Query_ = Query()
            donations = self.donations.search(Query_.milestone_id == milestone_id)
            for donation in donations:
                nft_id = len(self.nfts.all()) + 1
                self.nfts.insert({
                    'nft_id': nft_id,
                    'owner': donation['donor'],
                    'milestone_id': milestone_id,
                    'donation_amount': donation['amount'],
                    'minted_at': datetime.now().isoformat(),
                    'token_uri': f"ipfs://proof-of-impact/{nft_id}",
                    'metadata': {
                        'name': f"Proof of Impact #{nft_id}",
                        'description': f"Verified contribution"
                    }
                })
    
    def get_all_milestones(self) -> List[Dict]:
        # NUCLEAR: Bypass all checks, go straight to TinyDB
        if DB_AVAILABLE and hasattr(self, 'db') and self.db is not None:
            try:
                # Get table reference fresh every time
                milestones_table = self.db.table('milestones')
                results = milestones_table.all()
                return results if results else []
            except Exception as e:
                Logger.log('ERROR', f'get_all_milestones failed: {e}')
                return []
        return []
    
    def get_milestone(self, milestone_id: int) -> Optional[Dict]:
        if self.db:
            Query_ = Query()
            return self.milestones.get(Query_.id == milestone_id)
        return None
    
    def get_donations_for_milestone(self, milestone_id: int) -> List[Dict]:
        if self.db:
            Query_ = Query()
            return self.donations.search(Query_.milestone_id == milestone_id)
        return []
    
    def get_ngo_profile(self, ngo_address: str) -> Dict:
        if self.db:
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
        if self.db:
            Query_ = Query()
            return self.nfts.search(Query_.owner == donor_address)
        return []
    
    def get_analytics_summary(self) -> Dict:
        if not self.db:
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
            'avg_confidence': np.mean([a['confidence'] for a in all_analytics]),
            'pass_rate': len([a for a in all_analytics if a['result'] == 'PASS']) / len(all_analytics) * 100
        }

# ============================================================================
# AI VERIFIER
# ============================================================================

class EnhancedAIVerifier:
    def __init__(self):
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
            except Exception as e:
                Logger.log('ERROR', f"YOLO load failed: {e}")
    
    def detect_objects(self, image_bytes: bytes, conf_threshold: float = 0.3) -> List[Dict]:
        if not self.yolo_model:
            return [
                {'class': 0, 'conf': 0.85, 'label': 'person', 'xyxy': [100, 100, 300, 400]},
                {'class': 62, 'conf': 0.72, 'label': 'construction', 'xyxy': [200, 150, 500, 450]}
            ]
        tmp_path = Config.EVIDENCE_DIR / 'tmp_detect.jpg'
        with open(tmp_path, 'wb') as f:
            f.write(image_bytes)
        results = self.yolo_model.predict(source=str(tmp_path), conf=conf_threshold, save=False, imgsz=640)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'xyxy': box.xyxy[0].tolist(),
                    'label': result.names[int(box.cls[0])]
                })
        return detections
    
    def get_annotated_image(self, image_bytes: bytes, detections: List[Dict]) -> bytes:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(img)
        for det in detections:
            xyxy = det['xyxy']
            draw.rectangle(xyxy, outline='red', width=3)
            text = f"{det['label']}: {det['conf']:.2f}"
            draw.text((xyxy[0], xyxy[1] - 20), text, fill='red')
        output = BytesIO()
        img.save(output, format='JPEG')
        return output.getvalue()
    
    def get_exif_data(self, image_bytes: bytes) -> Dict:
        try:
            img = Image.open(BytesIO(image_bytes))
            exif = img._getexif() or {}
            readable = {}
            for tag, val in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                readable[decoded] = str(val)[:100]
            return readable
        except Exception:
            return {}
    
    def error_level_analysis(self, image_bytes: bytes) -> float:
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            buffer = BytesIO()
            img.save(buffer, 'JPEG', quality=90)
            buffer.seek(0)
            recompressed = Image.open(buffer)
            ela = ImageChops.difference(img, recompressed)
            arr = np.asarray(ela).astype(np.float32)
            return float(arr.mean())
        except Exception:
            return 0.0
    
    def calculate_gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    def verify_gps_location(self, image_bytes: bytes, expected_lat: float, expected_lon: float) -> Tuple[bool, str, Optional[float]]:
        exif = self.get_exif_data(image_bytes)
        if 'GPSInfo' not in exif:
            return False, "No GPS data", None
        distance = 100  # Mock distance
        if distance <= Config.GPS_RADIUS_METERS:
            return True, f"GPS verified ({distance:.0f}m)", distance
        return False, f"GPS too far ({distance:.0f}m)", distance
    
    def perceptual_hash_similarity(self, img_bytes_a: bytes, img_bytes_b: bytes) -> int:
        try:
            img_a = Image.open(BytesIO(img_bytes_a)).convert('RGB')
            img_b = Image.open(BytesIO(img_bytes_b)).convert('RGB')
            hash_a = imagehash.phash(img_a)
            hash_b = imagehash.phash(img_b)
            return int(hash_a - hash_b)
        except Exception:
            return 0
    
    def run_parallel_verification(self, evidence_bytes: bytes, before_bytes: Optional[bytes] = None,
                                  expected_lat: Optional[float] = None, expected_lon: Optional[float] = None,
                                  use_mock: bool = False) -> Dict:
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'pass': False,
            'confidence': 0.0
        }
        
        # GPS check
        exif = self.get_exif_data(evidence_bytes)
        if expected_lat and expected_lon:
            gps_ok, gps_msg, gps_dist = self.verify_gps_location(evidence_bytes, expected_lat, expected_lon)
        else:
            gps_ok = 'GPSInfo' in exif
            gps_msg = "GPS present" if gps_ok else "No GPS"
            gps_dist = None
        results['checks']['gps'] = {'passed': gps_ok, 'message': gps_msg, 'distance_meters': gps_dist}
        
        # Tamper
        tamper_score = self.error_level_analysis(evidence_bytes)
        tamper_ok = tamper_score < Config.TAMPER_THRESHOLD
        results['checks']['tamper'] = {'passed': tamper_ok, 'ela_score': round(tamper_score, 2)}
        
        # Objects
        detections = self.detect_objects(evidence_bytes)
        det_ok = len(detections) >= Config.MIN_DETECTIONS
        results['checks']['object_detection'] = {
            'passed': det_ok,
            'total_objects': len(detections),
            'top_detections': detections[:5]
        }
        
        # Before/after
        if before_bytes:
            phash_dist = self.perceptual_hash_similarity(before_bytes, evidence_bytes)
            comparison_ok = phash_dist > Config.PHASH_DISTANCE_THRESHOLD
            results['checks']['before_after'] = {
                'passed': comparison_ok,
                'phash_distance': phash_dist
            }
        else:
            comparison_ok = True
        
        # Confidence
        weights = {'gps': 0.25, 'tamper': 0.30, 'object_detection': 0.30, 'before_after': 0.15}
        confidence = sum(weights.get(k, 0) for k in results['checks'] if results['checks'][k]['passed'])
        
        results['confidence'] = min(1.0, round(confidence, 2))
        results['pass'] = results['confidence'] >= Config.CONFIDENCE_THRESHOLD
        results['annotated_image'] = self.get_annotated_image(evidence_bytes, detections)
        
        return results

# ============================================================================
# IPFS MANAGER
# ============================================================================

class EnhancedIPFSManager:
    @staticmethod
    def upload_to_web3storage(file_bytes: bytes, filename: str) -> Tuple[str, str]:
        token = Config.WEB3_STORAGE_TOKEN
        if token == 'DEMO_MODE':
            fake_cid = hashlib.sha256(file_bytes).hexdigest()[:46]
            return fake_cid, f"https://{fake_cid}.ipfs.dweb.link/{filename}"
        try:
            url = "https://api.web3.storage/upload"
            headers = {"Authorization": f"Bearer {token}"}
            files = {"file": (filename, file_bytes)}
            resp = requests.post(url, files=files, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cid = data['cid']
            return cid, f"https://{cid}.ipfs.dweb.link/{filename}"
        except Exception:
            fake_cid = hashlib.sha256(file_bytes).hexdigest()[:46]
            return fake_cid, f"local://{filename}"
    
    @staticmethod
    async def upload_async(file_bytes: bytes, filename: str) -> Tuple[str, str]:
        if not AIOHTTP_AVAILABLE:
            return EnhancedIPFSManager.upload_to_web3storage(file_bytes, filename)
        # Async implementation
        token = Config.WEB3_STORAGE_TOKEN
        if token == 'DEMO_MODE':
            await asyncio.sleep(0.1)
            fake_cid = hashlib.sha256(file_bytes).hexdigest()[:46]
            return fake_cid, f"https://{fake_cid}.ipfs.dweb.link/{filename}"
        return EnhancedIPFSManager.upload_to_web3storage(file_bytes, filename)
    
    @staticmethod
    def generate_qr_code(cid: str) -> Optional[bytes]:
        if not QRCODE_AVAILABLE:
            return None
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(f"ipfs://{cid}")
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
        except Exception:
            return None

# ============================================================================
# ASYNC MANAGER
# ============================================================================

class AsyncIOManager:
    @staticmethod
    async def verify_parallel_async(verifier, evidence_bytes: bytes, before_bytes: Optional[bytes],
                                   expected_lat: Optional[float], expected_lon: Optional[float]) -> Dict:
        # Simplified async version
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
    @staticmethod
    def create_confidence_gauge(confidence: float) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "AI Confidence"},
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
                modules.append(check_name.replace('_', ' ').title())
                actual = weight if checks[check_name]['passed'] else 0
                contributions.append(actual * 100 / confidence if confidence > 0 else 0)
                colors.append('green' if checks[check_name]['passed'] else 'red')
        fig = go.Figure(data=[go.Barpolar(r=contributions, theta=modules, marker=dict(color=colors))])
        fig.update_layout(title="AI Breakdown", height=400)
        return fig
    
    @staticmethod
    def create_donations_over_time(donations: List[Dict]) -> go.Figure:
        by_date = {}
        for don in donations:
            date = don['timestamp'][:10]
            if date not in by_date:
                by_date[date] = 0
            by_date[date] += don['amount'] / 10**18
        dates = sorted(by_date.keys())
        amounts = [by_date[d] for d in dates]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=amounts, mode='lines+markers', name='Daily'))
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
                'Status': m['status']
            })
        chart = alt.Chart(alt.Data(values=timeline_data)).mark_circle(size=200).encode(
            x='Date:T', y='Milestone:N', color='Status:N', tooltip=['Milestone', 'Status']
        ).properties(height=400).interactive()
        return chart

# ============================================================================
# ROLE MANAGER
# ============================================================================

class RoleManager:
    ROLES = ['NGO', 'Donor', 'Oracle', 'Guest']
    
    @staticmethod
    def login_page():
        if not STREAMLIT_AVAILABLE:
            return
        if 'user_role' not in st.session_state:
            st.session_state.user_role = 'Guest'
            st.session_state.user_id = 'demo_user'
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔐 User Role")
        selected = st.sidebar.selectbox("Select Role", RoleManager.ROLES,
                                       index=RoleManager.ROLES.index(st.session_state.user_role))
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
            st.warning(f"Access denied. Requires: {', '.join(allowed_roles)}")
            return False
        return True

# ============================================================================
# AI ASSISTANT
# ============================================================================

class AIAssistant:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.available = self.api_key and self.api_key != 'DEMO_MODE'
    
    def query(self, question: str, context: Dict) -> str:
        # Mock responses
        q = question.lower()
        if 'verified' in q:
            return f"✅ Project verified with {context.get('confidence', 85)}% confidence."
        elif 'status' in q:
            return f"📈 Status: {context.get('status', 'PENDING')}"
        else:
            return f"🤖 CharityChain AI Assistant. Total Projects: {context.get('total_projects', 0)}"

# ============================================================================
# BADGE SYSTEM
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
        """Check which badges a user has earned"""
        earned = []
        
        # Get user's donations and NFTs
        donations = [d for d in (db.donations.all() if db.db else []) if d['donor'] == user_id]
        nfts = db.get_donor_nfts(user_id)
        
        # Calculate metrics
        donation_count = len(donations)
        total_eth = sum(d['amount'] for d in donations) / 10**18 if donations else 0
        nft_count = len(nfts)
        
        # Check badge criteria
        if donation_count >= 5:
            earned.append('top_donor')
        
        if donation_count >= 1:
            earned.append('early_supporter')
        
        if total_eth >= 10:
            earned.append('impact_champion')
        
        if nft_count >= 5:
            earned.append('verified_100')
        
        # Check if user is an NGO
        milestones = db.get_all_milestones()
        user_milestones = [m for m in milestones if m['owner'] == user_id]
        completed = len([m for m in user_milestones if m['status'] == 'RELEASED'])
        
        if completed >= 10:
            earned.append('trusted_ngo')
        
        return earned
    
    @staticmethod
    def display_badges(badges: List[str]):
        """Display earned badges in Streamlit UI"""
        if not STREAMLIT_AVAILABLE:
            return
        
        if not badges:
            st.info("No badges earned yet. Keep contributing to earn achievements!")
            return
        
        st.subheader("🏅 Your Badges")
        
        # Display in columns (max 4 per row)
        cols = st.columns(min(len(badges), 4))
        
        for i, badge_id in enumerate(badges):
            badge_info = BadgeSystem.BADGES.get(badge_id, {})
            with cols[i % 4]:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 10px; color: white; margin-bottom: 1rem;'>
                    <h3>{badge_info.get('name', badge_id)}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def get_badge_progress(user_id: str, db: DatabaseManager) -> Dict[str, Dict]:
        """Get progress towards each badge"""
        donations = [d for d in (db.donations.all() if db.db else []) if d['donor'] == user_id]
        nfts = db.get_donor_nfts(user_id)
        milestones = db.get_all_milestones()
        user_milestones = [m for m in milestones if m['owner'] == user_id]
        
        progress = {}
        
        # Top Donor progress
        donation_count = len(donations)
        progress['top_donor'] = {
            'current': donation_count,
            'target': 5,
            'percentage': min(100, (donation_count / 5) * 100),
            'earned': donation_count >= 5
        }
        
        # Early Supporter progress
        progress['early_supporter'] = {
            'current': donation_count,
            'target': 1,
            'percentage': 100 if donation_count >= 1 else 0,
            'earned': donation_count >= 1
        }
        
        # Impact Champion progress
        total_eth = sum(d['amount'] for d in donations) / 10**18 if donations else 0
        progress['impact_champion'] = {
            'current': total_eth,
            'target': 10,
            'percentage': min(100, (total_eth / 10) * 100),
            'earned': total_eth >= 10
        }
        
        # Verified 100% progress
        nft_count = len(nfts)
        progress['verified_100'] = {
            'current': nft_count,
            'target': 5,
            'percentage': min(100, (nft_count / 5) * 100),
            'earned': nft_count >= 5
        }
        
        # Trusted NGO progress
        completed = len([m for m in user_milestones if m['status'] == 'RELEASED'])
        progress['trusted_ngo'] = {
            'current': completed,
            'target': 10,
            'percentage': min(100, (completed / 10) * 100),
            'earned': completed >= 10
        }
        
        return progress

# ============================================================================
# UI HELPER FUNCTIONS (For Streamlit Integration)
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling to Streamlit app"""
    if not STREAMLIT_AVAILABLE:
        return
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.5rem;
    }
    
    .metric-card p {
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    .metric-card small {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize Streamlit session state variables"""
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
    """Add dark mode toggle to sidebar"""
    if not STREAMLIT_AVAILABLE:
        return
    
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

def create_project_map(milestones: List[Dict]):
    """Create an interactive Folium map showing all project locations"""
    if not FOLIUM_AVAILABLE:
        return None
    
    if not milestones:
        return None
    
    # Calculate center of all points
    valid_milestones = [m for m in milestones if m.get('latitude', 0) != 0 and m.get('longitude', 0) != 0]
    
    if not valid_milestones:
        # Default to world center
        center_lat, center_lon = 20, 0
        zoom = 2
    else:
        center_lat = sum(m['latitude'] for m in valid_milestones) / len(valid_milestones)
        center_lon = sum(m['longitude'] for m in valid_milestones) / len(valid_milestones)
        zoom = 3
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='OpenStreetMap')
    
    # Add markers for each milestone
    for milestone in valid_milestones:
        lat = milestone['latitude']
        lon = milestone['longitude']
        
        # Determine marker color based on status
        status_colors = {
            'PENDING': 'blue',
            'SUBMITTED': 'orange',
            'VERIFIED': 'green',
            'REJECTED': 'red',
            'RELEASED': 'purple'
        }
        color = status_colors.get(milestone['status'], 'gray')
        
        # Create popup HTML
        popup_html = f"""
        <div style="width: 250px;">
            <h4>#{milestone['id']}: {milestone['description'][:50]}...</h4>
            <p><strong>NGO:</strong> {milestone['owner']}</p>
            <p><strong>Status:</strong> <span style="color: {color};">{milestone['status']}</span></p>
            <p><strong>Goal:</strong> {milestone['goal']/10**18:.2f} ETH</p>
            <p><strong>Raised:</strong> {milestone['raised']/10**18:.4f} ETH</p>
            <p><strong>Location:</strong> {lat:.4f}, {lon:.4f}</p>
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{milestone['id']}: {milestone['description'][:30]}...",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    return m

def show_nft_gallery(user_id: str, db: DatabaseManager):
    """Display NFT gallery for a user"""
    if not STREAMLIT_AVAILABLE:
        return
    
    nfts = db.get_donor_nfts(user_id)
    
    if not nfts:
        st.info("🎨 No NFTs yet. Donate to verified projects to earn Impact NFTs!")
        st.markdown("---")
        st.markdown("""
        ### How to Earn NFTs
        1. **Donate** to any charitable project
        2. Wait for **NGO to submit evidence**
        3. **Oracle verifies** the impact
        4. **NFT automatically minted** to your wallet!
        
        Each NFT represents verified real-world impact 🌍
        """)
        return
    
    st.success(f"🎉 You own **{len(nfts)}** Impact NFTs!")
    
    # Display badges
    badges = BadgeSystem.check_badges(user_id, db)
    if badges:
        st.markdown("---")
        BadgeSystem.display_badges(badges)
    
    st.markdown("---")
    st.subheader("🖼️ Your NFT Collection")
    
    # Display NFTs in a grid
    cols = st.columns(3)
    
    for i, nft in enumerate(nfts):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Impact NFT #{nft['nft_id']}</h3>
                <p><strong>Milestone:</strong> #{nft['milestone_id']}</p>
                <p><strong>Donation:</strong> {nft['donation_amount']/10**18:.4f} ETH</p>
                <p><strong>Minted:</strong> {nft['minted_at'][:10]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show IPFS link if available
            milestone = db.get_milestone(nft['milestone_id'])
            if milestone and 'evidence_cid' in milestone:
                st.markdown(f"[📸 View Evidence](https://w3s.link/ipfs/{milestone['evidence_cid']})")
    
    # Badge progress
    st.markdown("---")
    st.subheader("🎯 Badge Progress")
    
    progress_data = BadgeSystem.get_badge_progress(user_id, db)
    
    for badge_id, progress in progress_data.items():
        badge_info = BadgeSystem.BADGES.get(badge_id, {})
        badge_name = badge_info.get('name', badge_id)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{badge_name}**")
            st.progress(progress['percentage'] / 100)
        
        with col2:
            if progress['earned']:
                st.success("✅ Earned")
            else:
                st.caption(f"{progress['current']}/{progress['target']}")

def show_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    if not STREAMLIT_AVAILABLE:
        return
    
    st.header("📊 Platform Analytics")
    
    db = st.session_state.db
    
    # Get data
    milestones = db.get_all_milestones()
    donations = db.donations.all() if db.db else []
    verifications = db.verifications.all() if db.db else []
    nfts = db.nfts.all() if db.db else []
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", len(milestones))
    
    with col2:
        total_raised = sum(d['amount'] for d in donations) / 10**18
        st.metric("Total Raised", f"{total_raised:.2f} ETH")
    
    with col3:
        verified = len([m for m in milestones if m['status'] in ['VERIFIED', 'RELEASED']])
        st.metric("Verified Projects", verified)
    
    with col4:
        st.metric("NFTs Minted", len(nfts))
    
    st.markdown("---")
    
    # Visualizations
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
                status = m['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                hole=0.4
            )])
            fig.update_layout(title="Status Breakdown", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No project data yet")
    
    st.markdown("---")
    
    # Timeline
    st.subheader("📅 Project Timeline")
    if milestones:
        try:
            chart = VisualizationManager.create_timeline_chart(milestones)
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Timeline chart unavailable: {e}")
    else:
        st.info("No timeline data yet")
    
    st.markdown("---")
    
    # Verification stats
    st.subheader("🤖 AI Verification Statistics")
    
    if verifications:
        # Extract confidence from result object and convert to percentage
        confidences = [v['result']['confidence'] * 100 for v in verifications]
        avg_confidence = sum(confidences) / len(confidences)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col2:
            passed = len([c for c in confidences if c >= Config.CONFIDENCE_THRESHOLD * 100])
            st.metric("Passed Verifications", passed)
        
        with col3:
            failed = len([c for c in confidences if c < Config.CONFIDENCE_THRESHOLD * 100])
            st.metric("Failed Verifications", failed)
        
        # Confidence distribution
        fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence %",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No verification data yet")
    
    st.markdown("---")
    
    # Top performers
    col_top1, col_top2 = st.columns(2)
    
    with col_top1:
        st.subheader("🏆 Top Donors")
        donor_totals = {}
        for d in donations:
            donor = d['donor']
            donor_totals[donor] = donor_totals.get(donor, 0) + d['amount']
        
        if donor_totals:
            top_donors = sorted(donor_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (donor, amount) in enumerate(top_donors, 1):
                st.markdown(f"**{i}. {donor}** - {amount/10**18:.4f} ETH")
        else:
            st.info("No donor data yet")
    
    with col_top2:
        st.subheader("⭐ Top NGOs")
        ngo_stats = {}
        for m in milestones:
            ngo = m['owner']
            if ngo not in ngo_stats:
                ngo_stats[ngo] = {'total_raised': 0, 'projects': 0}
            ngo_stats[ngo]['total_raised'] += m['raised']
            ngo_stats[ngo]['projects'] += 1
        
        if ngo_stats:
            top_ngos = sorted(ngo_stats.items(), key=lambda x: x[1]['total_raised'], reverse=True)[:5]
            for i, (ngo, stats) in enumerate(top_ngos, 1):
                st.markdown(f"**{i}. {ngo}** - {stats['total_raised']/10**18:.4f} ETH ({stats['projects']} projects)")
        else:
            st.info("No NGO data yet")

def generate_demo_data(force=False):
    """
    Insert demo data DIRECTLY into the active DatabaseManager
    instance used by Streamlit in st.session_state.
    """
    if not STREAMLIT_AVAILABLE:
        Logger.log("ERROR", "Streamlit not available")
        return False
    
    if "db" not in st.session_state:
        Logger.log("ERROR", "No DB found in session state")
        return False

    db = st.session_state.db   # 🔥 Write to active DB instance

    # Force reset (wipe tables)
    if force and db.db:
        db.db.drop_tables()
        db._init_tables()

    # If projects already exist and not forcing reset → skip
    current = db.get_all_milestones()
    if current and not force:
        Logger.log("INFO", "Demo already exists → skipping", {'count': len(current)})
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
            ngo,
            int(eth * 10**18),
            desc,
            lat,
            lon
        )
        
        # Add donations to first 3 projects
        if i == 0:
            db.add_donation(milestone_id, "Alice Donor", int(0.5 * 10**18))
            db.add_donation(milestone_id, "Bob Supporter", int(0.3 * 10**18))
            db.add_donation(milestone_id, "Carol Benefactor", int(0.2 * 10**18))
        elif i == 1:
            db.add_donation(milestone_id, "David Philanthropist", int(2.0 * 10**18))
            db.add_donation(milestone_id, "Eva Contributor", int(1.5 * 10**18))
        elif i == 2:
            db.add_donation(milestone_id, "Frank Supporter", int(0.8 * 10**18))

    Logger.log("INFO", "Demo data inserted successfully", {'projects': len(demo)})
    return True