"""
verification.py - AI Evidence Verification REST API Endpoints
=============================================================
Endpoints under /api/v1 for triggering asynchronous multi-indicator verification,
polling job status, querying report history, inspecting technical checks,
and retrieving diagnostic artifacts.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Path, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.verification_service import VerificationService
from backend.app.api.schemas import (
    StandardResponse,
    StartVerificationRequest,
    VerificationJobResponse,
    VerificationReportResponse,
    VerificationHistoryResponse
)

router = APIRouter(tags=["AI Evidence Verification"])


@router.post(
    "/evidence/{evidence_id}/verification",
    response_model=StandardResponse[VerificationJobResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start AI Evidence Verification Job"
)
def start_evidence_verification(
    evidence_id: str = Path(..., description="Target evidence UUID"),
    req: Optional[StartVerificationRequest] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Initiates an asynchronous multi-indicator AI verification job.
    Enforces authorization, checks idempotency, and dispatches to background queue.
    """
    job = VerificationService.submit_verification_job(
        db=db,
        evidence_id=evidence_id,
        user=current_user,
        req=req
    )
    return StandardResponse(
        data=job,
        correlation_id=correlation_id_ctx.get(),
        message=f"Verification job '{job.id}' queued with status '{job.status}'."
    )


@router.get(
    "/verification-jobs/{job_id}",
    response_model=StandardResponse[VerificationJobResponse],
    summary="Get Verification Job Status"
)
def get_verification_job_status(
    job_id: str = Path(..., description="Verification job UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves execution status, attempt counts, and report reference for a job."""
    job = VerificationService.get_job_status(db, job_id, user=current_user)
    return StandardResponse(
        data=job,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/evidence/{evidence_id}/verification/latest",
    response_model=StandardResponse[VerificationReportResponse],
    summary="Get Latest Verification Report"
)
def get_latest_verification_report(
    evidence_id: str = Path(..., description="Target evidence UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves the most recent authorized verification report for an evidence asset."""
    report = VerificationService.get_latest_report(db, evidence_id, user=current_user)
    return StandardResponse(
        data=report,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/evidence/{evidence_id}/verification/history",
    response_model=StandardResponse[VerificationHistoryResponse],
    summary="Get Verification Report History"
)
def get_verification_history(
    evidence_id: str = Path(..., description="Target evidence UUID"),
    limit: int = Query(default=10, ge=1, le=50, description="Max reports to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves historical verification runs without overwriting past assessments."""
    history = VerificationService.get_verification_history(
        db=db,
        evidence_id=evidence_id,
        user=current_user,
        limit=limit,
        offset=offset
    )
    return StandardResponse(
        data=history,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/verification-reports/{report_id}",
    response_model=StandardResponse[VerificationReportResponse],
    summary="Get Verification Report Details"
)
def get_verification_report_detail(
    report_id: str = Path(..., description="Verification report UUID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Retrieves complete multi-indicator report details including all individual check records."""
    report = VerificationService.get_report_detail(db, report_id, user=current_user)
    return StandardResponse(
        data=report,
        correlation_id=correlation_id_ctx.get()
    )


@router.get(
    "/verification-reports/{report_id}/artifacts/{artifact_name}",
    summary="Download Diagnostic Visualization Artifact"
)
def download_diagnostic_artifact(
    report_id: str = Path(..., description="Verification report UUID"),
    artifact_name: str = Path(..., description="Artifact image filename"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Returns diagnostic image binary (e.g. YOLO bounding box annotation, ELA difference map)."""
    file_bytes, mime = VerificationService.get_diagnostic_artifact(
        db=db,
        report_id=report_id,
        artifact_name=artifact_name,
        user=current_user
    )
    return Response(content=file_bytes, media_type=mime)

