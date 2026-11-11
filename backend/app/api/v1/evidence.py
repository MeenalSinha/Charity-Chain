"""
evidence.py - Evidence Management REST API Endpoints
=====================================================
Multi-file upload, multi-versioning, cryptographic integrity checks,
download links, and tamper-evident audit trail endpoints under /api/v1.
"""

from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, Query, Path, UploadFile, File, Form, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.evidence_service import EvidenceService
from backend.app.integrations.storage.local import _validate_magic_bytes
from backend.app.api.schemas import (
    StandardResponse,
    EvidenceFileResponse,
    EvidenceBatchUploadResponse,
    EvidenceIntegrityResponse,
    EvidenceVersionHistoryResponse,
    EvidenceAuditLogResponse,
    ArchiveEvidenceRequest
)

router = APIRouter(tags=["Evidence Management"])


def _stream_and_validate_upload(
    upload_file: UploadFile,
    max_bytes: int,
    allowed_mime_types: set
) -> Tuple[bytes, str, str, int]:
    """
    Progressively streams upload in bounded 64KB chunks.
    Validates magic bytes on initial chunk and calculates SHA-256 in stream.
    Rejects oversized files immediately to prevent unbounded memory consumption.
    """
    import hashlib
    hasher = hashlib.sha256()
    total_size = 0
    chunks = []
    chunk_size = 65536
    first_chunk = True
    declared_mime = upload_file.content_type or "application/octet-stream"

    while True:
        chunk = upload_file.file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise ValidationException(
                f"File '{upload_file.filename}' ({total_size / (1024*1024):.2f} MB) exceeds maximum allowed size ({max_bytes / (1024*1024):.1f} MB)."
            )
        if first_chunk:
            if declared_mime not in allowed_mime_types:
                raise ValidationException(
                    f"File '{upload_file.filename}' has unsupported MIME type '{declared_mime}'. Allowed: {allowed_mime_types}"
                )
            if not _validate_magic_bytes(chunk, declared_mime):
                raise ValidationException(
                    f"File '{upload_file.filename}' contents do not match declared MIME type '{declared_mime}' magic signature."
                )
            first_chunk = False

        hasher.update(chunk)
        chunks.append(chunk)

    if total_size == 0:
        raise ValidationException(f"Evidence file '{upload_file.filename}' cannot be empty.")

    file_bytes = b"".join(chunks)
    sha256_hash = hasher.hexdigest()
    return file_bytes, sha256_hash, declared_mime, total_size


@router.post(
    "/milestones/{milestone_id}/evidence",
    response_model=StandardResponse[EvidenceBatchUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Multi-File Evidence Upload"
)
def upload_milestone_evidence(
    milestone_id: str = Path(..., description="Target milestone UUID"),
    files: List[UploadFile] = File(..., description="Batch of evidence files to upload"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Uploads one or more evidence files for milestone verification.
    Enforces batch limits, magic bytes inspection, SHA-256 deduplication,
    bounded streaming chunks, and initializes the tamper-evident audit hash chain.
    """
    max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    file_tuples = []
    for f in files:
        file_bytes, _, mime_type, _ = _stream_and_validate_upload(
            upload_file=f,
            max_bytes=max_bytes,
            allowed_mime_types=settings.ALLOWED_MIME_TYPES
        )
        file_tuples.append((file_bytes, f.filename or "evidence", mime_type))

    batch_result = EvidenceService.upload_evidence_files(
        db=db,
        milestone_id=milestone_id,
        files=file_tuples,
        uploader_id=current_user.id
    )
    return StandardResponse(
        data=batch_result,
        correlation_id=correlation_id_ctx.get(),
        message=f"Successfully uploaded {batch_result.total_uploaded} evidence file(s)."
    )


@router.get(
    "/milestones/{milestone_id}/evidence",
    response_model=StandardResponse[List[EvidenceFileResponse]],
    summary="List Milestone Evidence Assets"
)
def list_milestone_evidence(
    milestone_id: str = Path(..., description="Target milestone UUID"),
    include_archived: bool = Query(default=False, description="Whether to include soft-archived evidence"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists all evidence files submitted for the specified milestone."""
    evidence_list = EvidenceService.list_milestone_evidence(
        db=db,
        milestone_id=milestone_id,
        include_archived=include_archived,
        user=current_user
    )
    return StandardResponse(
        data=evidence_list,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/evidence/{evidence_id}",
    response_model=StandardResponse[EvidenceFileResponse],
    summary="Get Evidence Asset Details"
)
def get_evidence_detail(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves metadata and status for a specific evidence asset."""
    detail = EvidenceService.get_evidence_detail(db, evidence_id, user=current_user)
    return StandardResponse(
        data=detail,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/evidence/{evidence_id}/download",
    response_model=StandardResponse[dict],
    summary="Get Evidence Download URL"
)
def get_evidence_download_url(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Authorizes caller and returns a time-bounded presigned URL (S3/MinIO)
    or local download URL, and logs an immutable DOWNLOADED audit event.
    """
    download_url, _, filename, mime_type = EvidenceService.get_evidence_download(
        db=db,
        evidence_id=evidence_id,
        user_id=current_user.id
    )
    return StandardResponse(
        data={
            "evidence_id": evidence_id,
            "filename": filename,
            "mime_type": mime_type,
            "download_url": download_url
        },
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/evidence/{evidence_id}/content",
    summary="Download Raw Evidence Binary Content"
)
def download_evidence_content(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Streams the binary content of the evidence asset in chunks."""
    _, file_bytes, filename, mime_type = EvidenceService.get_evidence_download(
        db=db,
        evidence_id=evidence_id,
        user_id=current_user.id
    )
    if file_bytes is None:
        # If cloud storage provider without direct bytes, fetch through storage provider
        from backend.app.integrations.storage.factory import get_storage_provider
        evidence = EvidenceService.get_evidence_detail(db, evidence_id, user=current_user)
        storage = get_storage_provider()
        file_bytes = storage.get_file(evidence.storage_key)

    def iter_bytes():
        chunk_size = 65536
        for i in range(0, len(file_bytes), chunk_size):
            yield file_bytes[i:i + chunk_size]

    return StreamingResponse(
        iter_bytes(),
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.post(
    "/evidence/{evidence_id}/versions",
    response_model=StandardResponse[EvidenceFileResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload New Evidence Version"
)
def upload_evidence_version(
    evidence_id: str = Path(..., description="Parent evidence asset UUID"),
    file: UploadFile = File(..., description="New version replacement file"),
    reason: Optional[str] = Form(default=None, description="Reason for replacement version"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Uploads a replacement version for an existing evidence asset.
    Maintains historical records and links the new version cryptographically.
    """
    max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    file_bytes, _, mime_type, _ = _stream_and_validate_upload(
        upload_file=file,
        max_bytes=max_bytes,
        allowed_mime_types=settings.ALLOWED_MIME_TYPES
    )
    new_version = EvidenceService.upload_evidence_version(
        db=db,
        parent_evidence_id=evidence_id,
        file_bytes=file_bytes,
        filename=file.filename or "replacement_evidence",
        mime_type=mime_type,
        uploader_id=current_user.id,
        reason=reason
    )
    return StandardResponse(
        data=new_version,
        correlation_id=correlation_id_ctx.get(),
        message=f"Created version {new_version.version_number} successfully."
    )


@router.get(
    "/evidence/{evidence_id}/versions",
    response_model=StandardResponse[EvidenceVersionHistoryResponse],
    summary="Get Version History"
)
def get_evidence_versions(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves full version lineage for an evidence asset with access control enforcement."""
    history = EvidenceService.get_version_history(db, evidence_id, user=current_user)
    return StandardResponse(
        data=history,
        correlation_id=correlation_id_ctx.get()
    )


@router.post(
    "/evidence/{evidence_id}/verify-integrity",
    response_model=StandardResponse[EvidenceIntegrityResponse],
    summary="Verify Cryptographic Content Integrity"
)
def verify_evidence_integrity(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Downloads stored bytes from storage backend, recalculates SHA-256 hash,
    and compares against recorded hash. Appends result to audit hash chain.
    """
    result = EvidenceService.verify_evidence_integrity(
        db=db,
        evidence_id=evidence_id,
        actor_id=current_user.id,
        user=current_user
    )
    return StandardResponse(
        data=result,
        correlation_id=correlation_id_ctx.get(),
        message="Integrity check completed." if result.is_valid else "INTEGRITY VIOLATION DETECTED!"
    )


@router.post(
    "/evidence/{evidence_id}/archive",
    response_model=StandardResponse[EvidenceFileResponse],
    summary="Soft-Archive Evidence Asset"
)
def archive_evidence(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    req: Optional[ArchiveEvidenceRequest] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Soft-archives an evidence file. Underlying file content is preserved.
    Appends an ARCHIVED event to the audit hash chain.
    """
    reason = req.reason if req else None
    result = EvidenceService.archive_evidence(
        db=db,
        evidence_id=evidence_id,
        user_id=current_user.id,
        reason=reason
    )
    return StandardResponse(
        data=result,
        correlation_id=correlation_id_ctx.get(),
        message="Evidence asset soft-archived."
    )


@router.get(
    "/evidence/{evidence_id}/audit-trail",
    response_model=StandardResponse[EvidenceAuditLogResponse],
    summary="Get Tamper-Evident Audit Trail"
)
def get_evidence_audit_trail(
    evidence_id: str = Path(..., description="Evidence asset UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Retrieves the complete audit trail and verifies that the SHA-256
    hash chain is continuous, valid, and un-tampered.
    """
    trail = EvidenceService.get_audit_trail(db, evidence_id, user=current_user)
    return StandardResponse(
        data=trail,
        correlation_id=correlation_id_ctx.get(),
        message="Audit chain verified intact." if trail.is_chain_intact else "AUDIT CHAIN INTEGRITY BREACHED!"
    )

