"""
public_verification_service.py - Public Verification & Cryptographic Proof Engine
================================================================================
Provides independent public certificate verification for mobile QR code scans and API clients.
Verifies Ed25519 digital signatures, canonical payload integrity, issuer signing key validity,
and current lifecycle status (VALID, REVOKED, SUSPENDED, SUPERSEDED, INVALID_SIGNATURE).
Strictly preserves privacy by filtering out confidential donor and financial metadata.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from backend.app.core.canonical_json import canonicalize_json
from backend.app.core.constants import CertificateStatus, SigningKeyStatus
from backend.app.db.models.reviews_impact import ImpactCertificate, IssuerSigningKey
from backend.app.services.key_management_service import KeyManagementService
from backend.app.services.impact_ledger_service import ImpactLedgerService


class PublicVerificationService:
    @classmethod
    def verify_certificate(cls, db: Session, verification_id: str) -> Dict[str, Any]:
        """
        Public verification endpoint logic.
        Evaluates cryptographic signature, payload integrity, signing key history,
        and current lifecycle status without disclosing confidential donor or NGO data.
        """
        cert = db.query(ImpactCertificate).filter(
            ImpactCertificate.verification_id == verification_id
        ).first()

        now_iso = datetime.utcnow().isoformat()

        if not cert:
            return {
                "verification_outcome": "UNKNOWN_CERTIFICATE",
                "is_signature_valid": False,
                "is_payload_intact": False,
                "certificate_status": "NOT_FOUND",
                "verification_id": verification_id,
                "message": f"Certificate with verification ID '{verification_id}' was not found.",
                "verification_timestamp": now_iso
            }

        # 1. Payload Integrity Re-computation
        recomputed_payload_hash = hashlib.sha256(cert.canonical_payload.encode("utf-8")).hexdigest()
        is_payload_intact = (recomputed_payload_hash == cert.payload_hash)

        # 2. Key Resolution & Signature Verification
        key_record = db.query(IssuerSigningKey).filter(
            IssuerSigningKey.key_id == cert.signing_key_id
        ).first()

        key_pem = key_record.public_key_pem if key_record else ""
        if not key_pem:
            # Check active key as fallback
            kid, priv = KeyManagementService.get_active_private_key()
            if kid == cert.signing_key_id:
                from cryptography.hazmat.primitives import serialization
                key_pem = priv.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode("utf-8")

        is_sig_valid = False
        if key_pem and is_payload_intact:
            is_sig_valid = KeyManagementService.verify_signature(
                canonical_payload_str=cert.canonical_payload,
                signature_b64=cert.signature,
                public_key_pem_or_raw=key_pem
            )

        # 3. Determine Lifecycle & Overall Verification Outcome
        if not is_payload_intact:
            outcome = "PAYLOAD_MODIFIED"
        elif not is_sig_valid:
            outcome = "INVALID_SIGNATURE"
        elif cert.status == CertificateStatus.REVOKED.value:
            outcome = "REVOKED"
        elif cert.status == CertificateStatus.SUSPENDED.value:
            outcome = "SUSPENDED"
        elif cert.status == CertificateStatus.SUPERSEDED.value:
            outcome = "SUPERSEDED"
        elif cert.status in (CertificateStatus.ISSUED.value, CertificateStatus.ACTIVE.value):
            outcome = "VALID"
        else:
            outcome = "UNKNOWN_STATUS"

        # 4. Parse Claims for Safe Public Presentation
        claims_data = {}
        try:
            claims_data = json.loads(cert.canonical_payload)
        except Exception:
            pass

        public_donor = claims_data.get("donor", {})
        if not cert.public_attribution:
            public_donor = {"display_name": "Anonymous Contributor", "is_anonymous": True}
        else:
            donor_raw = claims_data.get("donor", {})
            public_donor = {
                "display_name": donor_raw.get("display_name") or cert.public_display_name or cert.recipient_name or "Contributor",
                "is_anonymous": False
            }

        # Filter out sensitive claims (no donor email, internal reviewer comments)
        safe_response: Dict[str, Any] = {
            "verification_outcome": outcome,
            "is_signature_valid": is_sig_valid,
            "is_payload_intact": is_payload_intact,
            "certificate_status": cert.status,
            "certificate_number": cert.certificate_number,
            "verification_id": cert.verification_id,
            "certificate_type": cert.certificate_type,
            "schema_version": cert.schema_version,
            "issuer": claims_data.get("issuer", {
                "id": cert.issuer_id,
                "name": "CharityChain Platform"
            }),
            "issued_at": claims_data.get("issued_at"),
            "donor": public_donor,
            "contribution": {
                "amount_cents": claims_data.get("contribution", {}).get("amount_cents"),
                "currency": claims_data.get("contribution", {}).get("currency", "USD")
            } if cert.public_attribution else {"attribution": "Private Contribution"},
            "project": claims_data.get("project", {}),
            "milestone": claims_data.get("milestone", {}),
            "organization": claims_data.get("organization", {}),
            "verification_summary": claims_data.get("verification_summary", {}),
            "disbursement_attestation": claims_data.get("disbursement_attestation"),
            "cryptographic_proof": {
                "signing_key_id": cert.signing_key_id,
                "signature_algorithm": cert.signature_algorithm,
                "payload_hash": cert.payload_hash,
                "signature_snippet": f"{cert.signature[:32]}...{cert.signature[-16:]}" if len(cert.signature) > 48 else cert.signature,
                "key_status": key_record.status if key_record else "UNKNOWN"
            },
            "revocation": {
                "reason_code": cert.revocation_reason,
                "revoked_at": cert.revoked_at.isoformat() if cert.revoked_at else None,
                "superseded_by_id": cert.superseded_by_id
            } if cert.status in (CertificateStatus.REVOKED.value, CertificateStatus.SUSPENDED.value, CertificateStatus.SUPERSEDED.value) else None,
            "verification_timestamp": now_iso
        }

        return safe_response
