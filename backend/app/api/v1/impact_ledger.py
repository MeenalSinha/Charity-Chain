"""
impact_ledger.py - Public Verification & Proof-of-Impact Ledger REST API Router
==============================================================================
Exposes public endpoints for mobile QR code verification, cryptographic proof inspection,
public key discovery (JWKS), milestone stream integrity checking, and administrative checkpointing.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.constants import PlatformRole
from backend.app.core.permissions import get_current_user, require_platform_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.reviews_impact import (
    ImpactCertificate, ImpactLedgerStream, ImpactLedgerEvent, ImpactLedgerCheckpoint
)
from backend.app.services.public_verification_service import PublicVerificationService
from backend.app.services.impact_ledger_service import ImpactLedgerService
from backend.app.services.key_management_service import KeyManagementService
from backend.app.api.schemas import (
    PublicCertificateVerificationResponse, ImpactLedgerEventResponseSchema,
    ImpactLedgerVerificationReport, ImpactLedgerCheckpointResponse
)

router = APIRouter(tags=["Impact Ledger & Verification"])


@router.get("/public/certificates/{verification_id}", response_model=PublicCertificateVerificationResponse)
def verify_public_certificate(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """
    Public verification endpoint called by mobile QR code scanners and independent verifiers.
    Returns structured cryptographic validity, payload integrity, and current lifecycle status.
    """
    result = PublicVerificationService.verify_certificate(db=db, verification_id=verification_id)
    return result


@router.get("/public/certificates/{verification_id}/proof")
def get_certificate_cryptographic_proof(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves the complete cryptographic proof envelope (canonical claims, signature, key ID)."""
    cert = db.query(ImpactCertificate).filter(
        ImpactCertificate.verification_id == verification_id
    ).first()
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    return {
        "certificate_number": cert.certificate_number,
        "verification_id": cert.verification_id,
        "schema_version": cert.schema_version,
        "canonical_payload": cert.canonical_payload,
        "payload_hash": cert.payload_hash,
        "signature": cert.signature,
        "signature_algorithm": cert.signature_algorithm,
        "signing_key_id": cert.signing_key_id,
        "issuer_id": cert.issuer_id,
        "status": cert.status
    }


@router.get("/.well-known/jwks.json")
def get_jwks(db: Session = Depends(get_db)):
    """Standard RFC 7517 JSON Web Key Set (JWKS) discovery endpoint for public verification keys."""
    return KeyManagementService.get_jwks(db=db)


@router.get("/public/issuers/{issuer_id}/keys")
def get_issuer_keys(
    issuer_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves public verification keys associated with a specific issuer identifier."""
    return KeyManagementService.get_jwks(db=db)


@router.get("/impact-ledger/milestones/{milestone_id}", response_model=List[ImpactLedgerEventResponseSchema])
def get_milestone_impact_events(
    milestone_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves public event history recorded on a milestone's Impact Ledger stream."""
    stream_id = f"milestone-{milestone_id}"
    events = db.query(ImpactLedgerEvent).filter(
        ImpactLedgerEvent.stream_id == stream_id
    ).order_by(ImpactLedgerEvent.sequence_number.asc()).all()

    return events


@router.get("/impact-ledger/milestones/{milestone_id}/verify", response_model=ImpactLedgerVerificationReport)
def verify_milestone_ledger_chain(
    milestone_id: str,
    db: Session = Depends(get_db)
):
    """Mathematically verifies cryptographic hash-chain continuity for a milestone event stream."""
    stream_id = f"milestone-{milestone_id}"
    report = ImpactLedgerService.verify_stream_chain(db=db, stream_id=stream_id)
    return report


@router.get("/admin/impact-ledger/streams")
def list_impact_ledger_streams(
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Lists all active Impact Ledger streams."""
    streams = db.query(ImpactLedgerStream).all()
    return streams


@router.get("/admin/impact-ledger/streams/{stream_id}/verify", response_model=ImpactLedgerVerificationReport)
def admin_verify_ledger_stream(
    stream_id: str,
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Runs full cryptographic verification on any event stream."""
    report = ImpactLedgerService.verify_stream_chain(db=db, stream_id=stream_id)
    return report


@router.post("/admin/impact-ledger/checkpoints", response_model=ImpactLedgerCheckpointResponse)
def create_ledger_checkpoint(
    stream_id: str,
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative endpoint: Creates an Ed25519 digitally signed checkpoint of an event stream head."""
    try:
        checkpoint = ImpactLedgerService.create_checkpoint(db=db, stream_id=stream_id)
        return checkpoint
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

