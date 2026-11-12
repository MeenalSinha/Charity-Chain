"""
certificates.py - Impact Certificates REST API Router
=====================================================
Provides endpoints for donor certificate retrieval, authorized PDF downloads,
cryptographic verification, public QR verification, and administrative lifecycle management.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session

from backend.app.core.constants import PlatformRole, CertificateStatus
from backend.app.core.permissions import get_current_user, require_platform_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import Donation
from backend.app.db.models.reviews_impact import ImpactCertificate, CertificateStatusEvent
from backend.app.services.certificate_service import CertificateService
from backend.app.services.certificate_eligibility_service import CertificateEligibilityService
from backend.app.services.public_verification_service import PublicVerificationService
from backend.app.services.key_management_service import KeyManagementService
from backend.app.integrations.storage import get_storage_provider
from backend.app.api.schemas import (
    IssueCertificateRequest, CertificateStatusUpdateRequest, CertificateSupersedeRequest,
    ImpactCertificateResponseSchema, PublicCertificateVerificationResponse,
    EvaluateCertificateEligibilityRequest, CertificateEligibilityResponseSchema
)

router = APIRouter(tags=["Certificates & Public Verification"])


# ==============================================================================
# 1. DONOR & AUTHENTICATED CERTIFICATE ENDPOINTS
# ==============================================================================

@router.get("/certificates/my", response_model=List[ImpactCertificateResponseSchema])
def get_my_certificates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves all Proof-of-Impact certificates issued to the authenticated donor."""
    user_donations = db.query(Donation.id).filter(Donation.donor_id == current_user.id).all()
    donation_ids = [d[0] for d in user_donations]

    if not donation_ids:
        return []

    certs = db.query(ImpactCertificate).filter(
        ImpactCertificate.donation_id.in_(donation_ids)
    ).order_by(ImpactCertificate.created_at.desc()).all()

    return certs


@router.post("/certificates/evaluate-eligibility", response_model=CertificateEligibilityResponseSchema)
def evaluate_eligibility(
    payload: EvaluateCertificateEligibilityRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Evaluates multi-gate eligibility for a donation before certificate issuance."""
    donation = db.query(Donation).filter(Donation.id == payload.donation_id).first()
    if not donation:
        raise HTTPException(status_code=404, detail="Donation not found.")

    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, "PLATFORM_ADMIN", "ADMIN"):
        if donation.donor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You can only evaluate eligibility for your own donations.")

    result = CertificateEligibilityService.evaluate_eligibility(
        db=db,
        donation_id=payload.donation_id,
        certificate_type=payload.certificate_type or "CONTRIBUTION_IMPACT",
        include_disbursement_claim=bool(payload.include_disbursement_claim)
    )
    return result.model_dump()


@router.post("/certificates/issue", response_model=ImpactCertificateResponseSchema)
def issue_certificate(
    payload: IssueCertificateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Issues a cryptographically signed Proof-of-Impact Certificate for a qualifying donation.
    Evaluates all multi-gate eligibility rules before issuance.
    """
    donation = db.query(Donation).filter(Donation.id == payload.donation_id).first()
    if not donation:
        raise HTTPException(status_code=404, detail="Donation not found.")

    # Only donor or admin can request certificate issuance
    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, "PLATFORM_ADMIN", "ADMIN"):
        if donation.donor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You can only request certificates for your own donations.")

    try:
        cert = CertificateService.issue_contribution_certificate(
            db=db,
            donation_id=payload.donation_id,
            actor_id=current_user.id,
            public_attribution=payload.public_attribution,
            public_display_name=payload.public_display_name,
            include_disbursement_claim=payload.include_disbursement_claim
        )
        return cert
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/certificates/{certificate_id}", response_model=ImpactCertificateResponseSchema)
def get_certificate_detail(
    certificate_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full details of a specific certificate with IDOR ownership validation."""
    cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == certificate_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, "PLATFORM_ADMIN", "ADMIN"):
        donation = db.query(Donation).filter(Donation.id == cert.donation_id).first()
        if not donation or donation.donor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this certificate.")

    return cert


@router.get("/certificates/{certificate_id}/download")
def download_certificate_pdf(
    certificate_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Downloads the official signed PDF certificate artifact."""
    cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == certificate_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, "PLATFORM_ADMIN", "ADMIN"):
        donation = db.query(Donation).filter(Donation.id == cert.donation_id).first()
        if not donation or donation.donor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this certificate.")

    # Retrieve PDF content
    pdf_bytes = None
    if cert.pdf_storage_key:
        try:
            storage = get_storage_provider()
            pdf_bytes = storage.read(cert.pdf_storage_key)
        except Exception:
            pass

    if not pdf_bytes and cert.pdf_storage_key:
        from pathlib import Path
        p = Path(cert.pdf_storage_key)
        if p.exists():
            with open(p, "rb") as f:
                pdf_bytes = f.read()

    if not pdf_bytes:
        from backend.app.services.certificate_rendering_service import CertificateRenderingService
        claims = json.loads(cert.canonical_payload) if cert.canonical_payload else {}
        verification_url = cert.qr_code_url or f"http://localhost:8000/api/v1/public/certificates/{cert.verification_id}"
        pdf_bytes, _ = CertificateRenderingService.render_certificate_pdf(
            claims=claims,
            signature_b64=cert.signature,
            signing_key_id=cert.signing_key_id,
            verification_url=verification_url
        )

    filename = f"{cert.certificate_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/certificates/{certificate_id}/history")
def get_certificate_history(
    certificate_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves the complete lifecycle audit event history for a certificate."""
    cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == certificate_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    if current_user.role not in (PlatformRole.PLATFORM_ADMIN, "PLATFORM_ADMIN", "ADMIN"):
        donation = db.query(Donation).filter(Donation.id == cert.donation_id).first()
        if not donation or donation.donor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this certificate history.")

    events = db.query(CertificateStatusEvent).filter(
        CertificateStatusEvent.certificate_id == certificate_id
    ).order_by(CertificateStatusEvent.created_at.asc()).all()

    return events


# ==============================================================================
# 2. ADMINISTRATIVE LIFECYCLE MANAGEMENT (SUSPEND / REVOKE / SUPERSEDE)
# ==============================================================================

@router.post("/certificates/{certificate_id}/suspend", response_model=ImpactCertificateResponseSchema)
def suspend_certificate(
    certificate_id: str,
    payload: CertificateStatusUpdateRequest,
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Temporarily suspends a certificate."""
    try:
        cert = CertificateService.suspend_certificate(
            db=db,
            certificate_id=certificate_id,
            reason_code=payload.reason_code,
            reason_description=payload.reason_description,
            actor_id=current_user.id
        )
        return cert
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/certificates/{certificate_id}/revoke", response_model=ImpactCertificateResponseSchema)
def revoke_certificate(
    certificate_id: str,
    payload: CertificateStatusUpdateRequest,
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Permanently revokes a certificate under governance policy."""
    try:
        cert = CertificateService.revoke_certificate(
            db=db,
            certificate_id=certificate_id,
            reason_code=payload.reason_code,
            reason_description=payload.reason_description,
            actor_id=current_user.id
        )
        return cert
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/certificates/{certificate_id}/supersede", response_model=ImpactCertificateResponseSchema)
def supersede_certificate(
    certificate_id: str,
    payload: CertificateSupersedeRequest,
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Marks an older certificate as superseded by a newer revision."""
    try:
        cert = CertificateService.supersede_certificate(
            db=db,
            old_certificate_id=certificate_id,
            new_certificate_id=payload.new_certificate_id,
            reason_code=payload.reason_code or "SUPERSEDED_BY_REVISION",
            reason_description=payload.reason_description,
            actor_id=current_user.id
        )
        return cert
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==============================================================================
# 3. PUBLIC VERIFICATION & PROOF ENDPOINTS (NO AUTH REQUIRED)
# ==============================================================================

@router.get("/public/certificates/{verification_id}", response_model=PublicCertificateVerificationResponse)
@router.get("/certificates/public/{verification_id}", response_model=PublicCertificateVerificationResponse)
def get_public_certificate_verification(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """
    Public QR-code verification endpoint.
    Performs full cryptographic validation and returns safe public claims.
    """
    result = PublicVerificationService.verify_certificate(db=db, verification_id=verification_id)
    return result


@router.get("/public/certificates/{verification_id}/proof")
def get_public_certificate_proof(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """
    Public endpoint returning the standalone cryptographic proof envelope
    for independent offline verification by external systems and auditors.
    """
    cert = db.query(ImpactCertificate).filter(ImpactCertificate.verification_id == verification_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail=f"Certificate '{verification_id}' not found.")

    return {
        "verification_id": cert.verification_id,
        "certificate_number": cert.certificate_number,
        "certificate_type": cert.certificate_type,
        "schema_version": cert.schema_version,
        "signature_algorithm": cert.signature_algorithm,
        "signing_key_id": cert.signing_key_id,
        "payload_hash": cert.payload_hash,
        "canonical_payload": cert.canonical_payload,
        "signature": cert.signature,
        "jwks_uri": "/.well-known/jwks.json",
        "status": cert.status
    }


@router.get("/public/certificates/{verification_id}/download")
def download_public_certificate_pdf(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """Public PDF download for verified certificates."""
    cert = db.query(ImpactCertificate).filter(ImpactCertificate.verification_id == verification_id).first()
    if not cert:
        raise HTTPException(status_code=404, detail=f"Certificate '{verification_id}' not found.")

    pdf_bytes = None
    if cert.pdf_storage_key:
        try:
            storage = get_storage_provider()
            pdf_bytes = storage.read(cert.pdf_storage_key)
        except Exception:
            pass

    if not pdf_bytes and cert.pdf_storage_key:
        from pathlib import Path
        p = Path(cert.pdf_storage_key)
        if p.exists():
            with open(p, "rb") as f:
                pdf_bytes = f.read()

    if not pdf_bytes:
        from backend.app.services.certificate_rendering_service import CertificateRenderingService
        claims = json.loads(cert.canonical_payload) if cert.canonical_payload else {}
        verification_url = cert.qr_code_url or f"http://localhost:8000/api/v1/public/certificates/{cert.verification_id}"
        pdf_bytes, _ = CertificateRenderingService.render_certificate_pdf(
            claims=claims,
            signature_b64=cert.signature,
            signing_key_id=cert.signing_key_id,
            verification_url=verification_url
        )

    filename = f"{cert.certificate_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ==============================================================================
# 4. RFC 7517 JWKS PUBLIC KEY DISCOVERY
# ==============================================================================

@router.get("/.well-known/jwks.json")
def get_jwks_discovery(db: Session = Depends(get_db)):
    """RFC 7517 JSON Web Key Set discovery endpoint for verifying Ed25519 signatures."""
    return KeyManagementService.build_jwks(db)
