"""
key_management_service.py - Ed25519 Cryptographic Signing & Public Key Registry
================================================================================
Manages asymmetric Ed25519 key lifecycle, secure signing, independent verification,
public key discovery (JWKS), and key rotation according to Phase 9 specifications.
"""

import os
import json
import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import SigningKeyStatus
from backend.app.db.models.reviews_impact import IssuerSigningKey

_override_private_key: Optional[ed25519.Ed25519PrivateKey] = None
_override_key_id: Optional[str] = None


class KeyManagementService:
    @staticmethod
    def set_key_override(private_key: Optional[ed25519.Ed25519PrivateKey], key_id: Optional[str] = None) -> None:
        """Sets an in-memory key override (used during unit testing)."""
        global _override_private_key, _override_key_id
        _override_private_key = private_key
        _override_key_id = key_id

    @staticmethod
    def generate_keypair(key_id: Optional[str] = None) -> Tuple[ed25519.Ed25519PrivateKey, str, str, Dict[str, Any]]:
        """
        Generates a new Ed25519 asymmetric keypair.
        Returns: (private_key, private_pem, public_pem, jwk_dict)
        """
        kid = key_id or f"key-charitychain-{int(datetime.utcnow().timestamp())}"
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode("utf-8")

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode("utf-8")

        raw_public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        x_b64 = base64.urlsafe_b64encode(raw_public_bytes).decode("utf-8").rstrip("=")

        jwk_dict = {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": x_b64,
            "kid": kid,
            "use": "sig",
            "alg": "EdDSA"
        }

        return private_key, private_pem, public_pem, jwk_dict

    @classmethod
    def get_active_private_key(cls) -> Tuple[str, ed25519.Ed25519PrivateKey]:
        """
        Retrieves the active Ed25519 private signing key and key identifier.
        Supports in-memory overrides, environment variables, or local development files.
        """
        global _override_private_key, _override_key_id
        if _override_private_key is not None:
            kid = _override_key_id or getattr(settings, "SIGNING_ACTIVE_KEY_ID", "key-charitychain-test")
            return kid, _override_private_key

        kid = getattr(settings, "SIGNING_ACTIVE_KEY_ID", "key-charitychain-2026-v1")

        # 1. Check environment variable
        b64_key = getattr(settings, "SIGNING_PRIVATE_KEY_B64", None)
        if b64_key:
            key_bytes = base64.b64decode(b64_key)
            try:
                private_key = serialization.load_pem_private_key(key_bytes, password=None)
                if isinstance(private_key, ed25519.Ed25519PrivateKey):
                    return kid, private_key
            except Exception:
                pass
            if len(key_bytes) == 32:
                return kid, ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)

        # 2. Check local key directory
        key_dir = Path(getattr(settings, "SIGNING_KEY_DIR", "data/keys"))
        key_dir.mkdir(parents=True, exist_ok=True)
        key_file = key_dir / f"{kid}.pem"

        if key_file.exists():
            with open(key_file, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
                if isinstance(private_key, ed25519.Ed25519PrivateKey):
                    return kid, private_key

        # 3. In dev/testing, auto-provision and save local key
        if getattr(settings, "ENVIRONMENT", "development") in ("development", "testing"):
            private_key, private_pem, _, _ = cls.generate_keypair(key_id=kid)
            with open(key_file, "w", encoding="utf-8") as f:
                f.write(private_pem)
            return kid, private_key

        raise ValueError(
            f"Active signing key '{kid}' not found! Production requires an explicitly provisioned key."
        )

    @classmethod
    def sign_payload(cls, canonical_payload_str: str, key_id: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Signs a canonical payload string using the active Ed25519 private key.
        Returns: (signature_b64, key_id, payload_hash)
        """
        active_kid, private_key = cls.get_active_private_key()
        kid = key_id or active_kid

        payload_bytes = canonical_payload_str.encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        # Ed25519 signs the canonical payload bytes directly
        signature_bytes = private_key.sign(payload_bytes)
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

        return signature_b64, kid, payload_hash

    @staticmethod
    def verify_signature(canonical_payload_str: str, signature_b64: str, public_key_pem_or_raw: str) -> bool:
        """
        Independently verifies an Ed25519 signature against canonical payload string.
        """
        try:
            signature_bytes = base64.b64decode(signature_b64)
            payload_bytes = canonical_payload_str.encode("utf-8")

            # Load public key
            if "BEGIN PUBLIC KEY" in public_key_pem_or_raw:
                public_key = serialization.load_pem_public_key(public_key_pem_or_raw.encode("utf-8"))
            else:
                raw_bytes = base64.urlsafe_b64decode(public_key_pem_or_raw + "==")
                public_key = ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)

            if not isinstance(public_key, ed25519.Ed25519PublicKey):
                return False

            public_key.verify(signature_bytes, payload_bytes)
            return True
        except (InvalidSignature, ValueError, Exception):
            return False

    @staticmethod
    def register_public_key(
        db: Session,
        key_id: str,
        public_key_pem: str,
        public_key_jwk: str,
        status: str = SigningKeyStatus.ACTIVE.value
    ) -> IssuerSigningKey:
        """Registers a public verification key and its lifecycle metadata in the database."""
        existing = db.query(IssuerSigningKey).filter(IssuerSigningKey.key_id == key_id).first()
        if existing:
            existing.public_key_pem = public_key_pem
            existing.public_key_jwk = public_key_jwk
            existing.status = status
            db.commit()
            db.refresh(existing)
            return existing

        record = IssuerSigningKey(
            key_id=key_id,
            algorithm="Ed25519",
            public_key_pem=public_key_pem,
            public_key_jwk=public_key_jwk,
            status=status,
            usage="SIG",
            activated_at=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @classmethod
    def ensure_active_key_registered(cls, db: Session) -> IssuerSigningKey:
        """Ensures the currently active signing key's public component is registered in the database."""
        kid, private_key = cls.get_active_private_key()
        existing = db.query(IssuerSigningKey).filter(IssuerSigningKey.key_id == kid).first()
        if existing:
            return existing

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode("utf-8")
        raw_public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        x_b64 = base64.urlsafe_b64encode(raw_public_bytes).decode("utf-8").rstrip("=")
        jwk_dict = {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": x_b64,
            "kid": kid,
            "use": "sig",
            "alg": "EdDSA"
        }
        return cls.register_public_key(
            db=db,
            key_id=kid,
            public_key_pem=public_pem,
            public_key_jwk=json.dumps(jwk_dict),
            status=SigningKeyStatus.ACTIVE.value
        )

    @staticmethod
    def get_jwks(db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Generates RFC 7517 compliant JSON Web Key Set (JWKS) for public key discovery.
        """
        keys: List[Dict[str, Any]] = []
        if db is not None:
            records = db.query(IssuerSigningKey).filter(
                IssuerSigningKey.status.in_([SigningKeyStatus.ACTIVE.value, SigningKeyStatus.RETIRED.value])
            ).all()
            for r in records:
                try:
                    jwk_obj = json.loads(r.public_key_jwk)
                    keys.append(jwk_obj)
                except Exception:
                    pass

        if not keys:
            # Fallback to active key directly
            kid, private_key = KeyManagementService.get_active_private_key()
            public_key = private_key.public_key()
            raw_public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            x_b64 = base64.urlsafe_b64encode(raw_public_bytes).decode("utf-8").rstrip("=")
            keys.append({
                "kty": "OKP",
                "crv": "Ed25519",
                "x": x_b64,
                "kid": kid,
                "use": "sig",
                "alg": "EdDSA"
            })

        return {"keys": keys}

    build_jwks = get_jwks

    @classmethod
    def rotate_key(cls, db: Session, new_key_id: str) -> Tuple[str, IssuerSigningKey]:
        """
        Rotates the active key: retires the current key in the database and registers the new key.
        """
        current_kid, _ = cls.get_active_private_key()
        current_record = db.query(IssuerSigningKey).filter(IssuerSigningKey.key_id == current_kid).first()
        if current_record and current_record.status == SigningKeyStatus.ACTIVE.value:
            current_record.status = SigningKeyStatus.RETIRED.value
            current_record.retired_at = datetime.utcnow()
            db.commit()

        # Generate new keypair
        new_private, new_private_pem, new_public_pem, new_jwk = cls.generate_keypair(key_id=new_key_id)
        key_dir = Path(getattr(settings, "SIGNING_KEY_DIR", "data/keys"))
        key_dir.mkdir(parents=True, exist_ok=True)
        with open(key_dir / f"{new_key_id}.pem", "w", encoding="utf-8") as f:
            f.write(new_private_pem)

        new_record = cls.register_public_key(
            db=db,
            key_id=new_key_id,
            public_key_pem=new_public_pem,
            public_key_jwk=json.dumps(new_jwk),
            status=SigningKeyStatus.ACTIVE.value
        )
        return new_key_id, new_record
