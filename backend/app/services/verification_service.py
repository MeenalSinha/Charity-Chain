"""
verification_service.py - Domain Service for AI Evidence Verification
=====================================================================
Orchestrates multi-indicator AI verification, durable job queues,
background task execution, historical report persistence, cryptographic
audit trail integration, and role-based access control.
"""

import os
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import (
    VerificationCheckStatus, OverallAssessmentStatus, VerificationJobStatus
)
from backend.app.core.exceptions import (
    NotFoundException, ValidationException, ConflictException,
    PermissionDeniedError, StorageException
)
from backend.app.core.logging import logger
from backend.app.db.session import SessionFactory
from backend.app.db.models.projects import Milestone, Project, ProjectLocation
from backend.app.db.models.users import User
from backend.app.db.models.evidence import (
    EvidenceFile, VerificationReport, VerificationCheckRecord,
    VerificationJob, EvidenceAuditEvent, compute_audit_hash
)
from backend.app.integrations.storage.factory import get_storage_provider
from backend.app.integrations.ai.factory import get_ai_provider
from backend.app.integrations.ai.schemas import VerificationOutcome
from backend.app.services.evidence_service import EvidenceService
from backend.app.api.schemas import (
    StartVerificationRequest, VerificationJobResponse,
    VerificationReportResponse, VerificationCheckResponse,
    VerificationHistoryResponse
)


class VerificationService:
    """Domain service managing the complete multi-indicator AI verification lifecycle."""

    @classmethod
    def submit_verification_job(
        cls,
        db: Session,
        evidence_id: str,
        user: User,
        req: Optional[StartVerificationRequest] = None
    ) -> VerificationJobResponse:
        """
        Validates authorization, checks idempotency, creates a durable VerificationJob,
        and triggers asynchronous verification processing.
        """
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence asset '{evidence_id}' not found.")

        # Access check: NGO staff, Reviewer, or Platform Admin can trigger verification
        EvidenceService._check_evidence_access(db, evidence, user, action="integrity")

        baseline_id = req.baseline_evidence_id if req else None
        idempotency_key = req.idempotency_key if req else None
        requested_checks = req.requested_checks if req else None

        # Check idempotency
        if idempotency_key:
            existing_job = db.execute(
                select(VerificationJob).where(VerificationJob.idempotency_key == idempotency_key)
            ).scalar_one_or_none()
            if existing_job:
                return cls._build_job_response(existing_job)

        # Validate baseline evidence if provided
        if baseline_id:
            baseline = db.execute(select(EvidenceFile).where(EvidenceFile.id == baseline_id)).scalar_one_or_none()
            if not baseline or baseline.is_archived:
                raise ValidationException(f"Baseline evidence asset '{baseline_id}' not found or is archived.")
            EvidenceService._check_evidence_access(db, baseline, user, action="read")

        # Create durable VerificationJob
        job = VerificationJob(
            id=str(uuid.uuid4()),
            evidence_file_id=evidence.id,
            evidence_hash=evidence.file_hash,
            evidence_version=evidence.version_number,
            requested_by_id=user.id,
            status=VerificationJobStatus.PENDING.value,
            attempt_count=0,
            max_attempts=settings.VERIFICATION_JOB_MAX_ATTEMPTS,
            pipeline_version=settings.VERIFICATION_PIPELINE_VERSION,
            requested_checks_json=requested_checks,
            baseline_evidence_id=baseline_id,
            idempotency_key=idempotency_key,
            queued_at=datetime.utcnow()
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Execute job (in development/sync mode or via worker)
        cls.process_job_worker(job.id, db=db)
        db.refresh(job)

        return cls._build_job_response(job)

    @classmethod
    def process_job_worker(cls, job_id: str, db: Optional[Session] = None) -> None:
        """
        Executes the verification pipeline for a queued job.
        Fetches stored bytes, validates integrity, runs all checks,
        persists VerificationReport and records audit log.
        """
        if db is not None:
            cls._execute_job(db, job_id)
        else:
            with SessionFactory() as new_db:
                cls._execute_job(new_db, job_id)

    @classmethod
    def _execute_job(cls, db: Session, job_id: str) -> None:
        job = db.execute(select(VerificationJob).where(VerificationJob.id == job_id)).scalar_one_or_none()
        if not job or job.status == VerificationJobStatus.COMPLETED.value:
            return

        job.status = VerificationJobStatus.PROCESSING.value
        job.started_at = datetime.utcnow()
        job.attempt_count += 1
        db.commit()

        try:
            evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == job.evidence_file_id)).scalar_one_or_none()
            if not evidence:
                raise NotFoundException("Target evidence asset missing.")

            milestone = db.execute(select(Milestone).where(Milestone.id == evidence.milestone_id)).scalar_one_or_none()
            project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none() if milestone else None

            # Retrieve stored evidence bytes
            storage = get_storage_provider()
            evidence_bytes = storage.get_file(evidence.storage_key)

            # Retrieve baseline bytes if requested
            baseline_bytes = None
            baseline_hash = None
            if job.baseline_evidence_id:
                baseline_file = db.execute(select(EvidenceFile).where(EvidenceFile.id == job.baseline_evidence_id)).scalar_one_or_none()
                if baseline_file:
                    baseline_bytes = storage.get_file(baseline_file.storage_key)
                    baseline_hash = baseline_file.file_hash

            # Resolve project / milestone coordinates if available
            expected_lat = None
            expected_lon = None
            loc = db.execute(select(ProjectLocation).where(ProjectLocation.milestone_id == milestone.id)).scalar_one_or_none() if milestone else None
            if loc:
                expected_lat = loc.latitude
                expected_lon = loc.longitude

            # Context for category-aware relevance and XAI summary
            context = {
                "project_title": project.title if project else "Project",
                "milestone_title": milestone.title if milestone else "Milestone",
                "project_description": project.description if project else "",
                "category_name": str(project.organization_id) if project else "",
                "expected_lat": expected_lat,
                "expected_lon": expected_lon,
                "mime_type": evidence.mime_type,
                "expected_sha256": evidence.file_hash,
                "generate_artifacts": True
            }

            # Run Multi-Indicator AI Provider
            ai_provider = get_ai_provider()
            outcome: VerificationOutcome = ai_provider.analyze_evidence_sync(
                evidence_bytes=evidence_bytes,
                mime_type=evidence.mime_type,
                expected_sha256=evidence.file_hash,
                baseline_bytes=baseline_bytes,
                baseline_evidence_id=job.baseline_evidence_id,
                baseline_hash=baseline_hash,
                context=context
            )

            # Persist VerificationReport
            report = VerificationReport(
                id=str(uuid.uuid4()),
                evidence_file_id=evidence.id,
                evidence_version=evidence.version_number,
                evidence_hash=evidence.file_hash,
                project_id=project.id if project else None,
                milestone_id=milestone.id if milestone else None,
                pipeline_version=outcome.pipeline_version,
                overall_assessment=outcome.overall_assessment.value,
                confidence_score=outcome.confidence_score,
                calibration_status=outcome.calibration_status,
                processing_started_at=job.started_at,
                processing_completed_at=datetime.utcnow(),
                provider_name=outcome.model_metadata.get("yolo_model", "yolo_forensics"),
                model_metadata_json=outcome.model_metadata,
                summary_notes=outcome.summary_notes,
                limitations_json=outcome.limitations,
                config_snapshot_json={
                    "confidence_threshold": settings.VERIFICATION_CONFIDENCE_THRESHOLD,
                    "tamper_threshold": settings.TAMPER_ELA_THRESHOLD,
                    "gps_radius": settings.GPS_RADIUS_METERS
                },
                overall_confidence=outcome.confidence_score or 0.0,
                overall_pass=(outcome.overall_assessment == OverallAssessmentStatus.CONSISTENT)
            )
            db.add(report)
            db.flush()

            # Persist VerificationCheckRecord items
            for check in outcome.checks:
                chk_status_val = check.status.value if hasattr(check.status, "value") else str(check.status)
                rec = VerificationCheckRecord(
                    id=str(uuid.uuid4()),
                    report_id=report.id,
                    check_type=check.check_type,
                    status=check.status,
                    passed=check.passed,
                    message=check.message,
                    technical_result_json=check.technical_result,
                    raw_metrics_json=check.raw_metrics,
                    model_name=check.model_name,
                    model_version=check.model_version,
                    duration_ms=check.duration_ms,
                    diagnostic_artifact_path=check.diagnostic_artifact_path,
                    details_json=check.message
                )
                db.add(rec)

            # Update VerificationJob
            job.status = VerificationJobStatus.COMPLETED.value
            job.report_id = report.id
            job.completed_at = datetime.utcnow()

            # Cryptographic audit log on evidence asset
            last_event = db.execute(
                select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == evidence.id).order_by(EvidenceAuditEvent.created_at.desc())
            ).scalars().first()
            prev_hash = last_event.current_event_hash if last_event else ("0" * 64)
            now_iso = datetime.utcnow().isoformat()
            details = {
                "action": "AI_VERIFICATION_PROCESSED",
                "report_id": report.id,
                "overall_assessment": report.overall_assessment,
                "confidence_score": report.confidence_score,
                "pipeline_version": report.pipeline_version,
                "timestamp": now_iso
            }
            curr_hash = compute_audit_hash(
                previous_hash=prev_hash,
                evidence_id=evidence.id,
                actor_id=job.requested_by_id,
                event_type="AI_VERIFIED",
                details=details
            )
            audit = EvidenceAuditEvent(
                evidence_id=evidence.id,
                actor_id=job.requested_by_id,
                event_type="AI_VERIFIED",
                details_json=details,
                previous_event_hash=prev_hash,
                current_event_hash=curr_hash
            )
            db.add(audit)
            db.commit()

        except Exception as e:
            logger.error(f"Verification job '{job_id}' failed: {e}", exc_info=True)
            job.error_category = "PIPELINE_EXECUTION_FAILURE"
            job.error_message = str(e)[:500]
            if job.attempt_count < job.max_attempts:
                job.status = VerificationJobStatus.RETRYING.value
            else:
                job.status = VerificationJobStatus.FAILED.value
            job.completed_at = datetime.utcnow()
            db.commit()

    @classmethod
    def get_job_status(cls, db: Session, job_id: str, user: User) -> VerificationJobResponse:
        """Retrieves status and report reference for a verification job."""
        job = db.execute(select(VerificationJob).where(VerificationJob.id == job_id)).scalar_one_or_none()
        if not job:
            raise NotFoundException(f"Verification job '{job_id}' not found.")

        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == job.evidence_file_id)).scalar_one_or_none()
        if evidence:
            EvidenceService._check_evidence_access(db, evidence, user, action="read")

        return cls._build_job_response(job)

    @classmethod
    def get_latest_report(cls, db: Session, evidence_id: str, user: User) -> VerificationReportResponse:
        """Retrieves the most recent authorized verification report for an evidence asset."""
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        EvidenceService._check_evidence_access(db, evidence, user, action="read")

        report = db.execute(
            select(VerificationReport)
            .where(VerificationReport.evidence_file_id == evidence.id)
            .order_by(VerificationReport.created_at.desc())
        ).scalars().first()

        if not report:
            raise NotFoundException(f"No verification report found for evidence '{evidence_id}'.")

        return cls._build_report_response(report)

    @classmethod
    def get_verification_history(
        cls,
        db: Session,
        evidence_id: str,
        user: User,
        limit: int = 10,
        offset: int = 0
    ) -> VerificationHistoryResponse:
        """Retrieves all historical verification reports for an evidence asset."""
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        EvidenceService._check_evidence_access(db, evidence, user, action="read")

        reports = db.execute(
            select(VerificationReport)
            .where(VerificationReport.evidence_file_id == evidence.id)
            .order_by(VerificationReport.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).scalars().all()

        return VerificationHistoryResponse(
            evidence_file_id=evidence.id,
            total_reports=len(reports),
            reports=[cls._build_report_response(r) for r in reports]
        )

    @classmethod
    def get_report_detail(cls, db: Session, report_id: str, user: User) -> VerificationReportResponse:
        """Retrieves complete multi-indicator report details with check records."""
        report = db.execute(select(VerificationReport).where(VerificationReport.id == report_id)).scalar_one_or_none()
        if not report:
            raise NotFoundException(f"Verification report '{report_id}' not found.")

        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == report.evidence_file_id)).scalar_one_or_none()
        if evidence:
            EvidenceService._check_evidence_access(db, evidence, user, action="read")

        return cls._build_report_response(report)

    @classmethod
    def get_diagnostic_artifact(
        cls,
        db: Session,
        report_id: str,
        artifact_name: str,
        user: User
    ) -> Tuple[bytes, str]:
        """
        Validates authorization and path containment, returning diagnostic artifact bytes.
        """
        report = db.execute(select(VerificationReport).where(VerificationReport.id == report_id)).scalar_one_or_none()
        if not report:
            raise NotFoundException(f"Verification report '{report_id}' not found.")

        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == report.evidence_file_id)).scalar_one_or_none()
        if evidence:
            EvidenceService._check_evidence_access(db, evidence, user, action="read")

        clean_name = os.path.basename(artifact_name)
        base_dir = Path(settings.DIAGNOSTIC_ARTIFACTS_DIR).resolve()
        target = (base_dir / clean_name).resolve()

        try:
            target.relative_to(base_dir)
        except ValueError:
            raise ValidationException("Path traversal attempt detected in artifact request.")

        if not target.exists() or not target.is_file():
            raise NotFoundException(f"Diagnostic artifact '{clean_name}' not found.")

        mime = "image/jpeg" if clean_name.lower().endswith(".jpg") else "image/png"
        return target.read_bytes(), mime

    @staticmethod
    def _build_job_response(job: VerificationJob) -> VerificationJobResponse:
        return VerificationJobResponse(
            id=job.id,
            evidence_file_id=job.evidence_file_id,
            evidence_hash=job.evidence_hash,
            evidence_version=job.evidence_version,
            status=job.status,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            pipeline_version=job.pipeline_version,
            baseline_evidence_id=job.baseline_evidence_id,
            report_id=job.report_id,
            error_category=job.error_category,
            error_message=job.error_message,
            queued_at=job.queued_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )

    @staticmethod
    def _build_report_response(report: VerificationReport) -> VerificationReportResponse:
        checks_schemas = []
        for c in (report.checks or []):
            artifact_url = None
            if c.diagnostic_artifact_path:
                art_name = os.path.basename(c.diagnostic_artifact_path)
                artifact_url = f"/api/v1/verification-reports/{report.id}/artifacts/{art_name}"

            checks_schemas.append(VerificationCheckResponse(
                id=c.id,
                check_type=c.check_type,
                status=c.status.value if hasattr(c.status, "value") else str(c.status),
                passed=c.passed,
                message=c.message,
                technical_result=c.technical_result_json,
                raw_metrics=c.raw_metrics_json,
                model_name=c.model_name,
                model_version=c.model_version,
                duration_ms=c.duration_ms,
                diagnostic_artifact_url=artifact_url,
                created_at=c.created_at
            ))

        return VerificationReportResponse(
            id=report.id,
            evidence_file_id=report.evidence_file_id,
            evidence_version=report.evidence_version,
            evidence_hash=report.evidence_hash,
            project_id=report.project_id,
            milestone_id=report.milestone_id,
            pipeline_version=report.pipeline_version,
            overall_assessment=report.overall_assessment,
            confidence_score=report.confidence_score,
            calibration_status=report.calibration_status,
            summary_notes=report.summary_notes,
            limitations=report.limitations_json,
            model_metadata=report.model_metadata_json,
            error_details=report.error_details,
            checks=checks_schemas,
            created_at=report.created_at,
            processing_started_at=report.processing_started_at,
            processing_completed_at=report.processing_completed_at
        )

