"""
evidence_service.py - Domain Service for Evidence Management
============================================================
Handles multi-file uploads, magic-byte validation, streaming SHA-256 deduplication,
storage dispatch (Local/S3/MinIO), multi-versioning, cryptographic audit hash chains,
and soft-archival without data loss.
"""

import hashlib
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import EvidenceStatus, OrganizationRole, UserRole
from backend.app.core.exceptions import (
    NotFoundException, ValidationException, ConflictException,
    PermissionDeniedError
)
from backend.app.db.models.projects import Milestone, Project
from backend.app.db.models.users import User, OrganizationMembership
from backend.app.db.models.evidence import (
    EvidenceFile, EvidenceAuditEvent, compute_audit_hash
)
from backend.app.integrations.storage.factory import get_storage_provider
from backend.app.integrations.storage.local import _validate_magic_bytes
from backend.app.api.schemas import (
    EvidenceFileResponse, EvidenceBatchUploadResponse,
    EvidenceIntegrityResponse, EvidenceAuditEventResponse,
    EvidenceAuditLogResponse, EvidenceVersionHistoryResponse
)


class EvidenceService:
    """Domain service managing milestone evidence lifecycle and integrity verification."""

    @staticmethod
    def _check_upload_permissions(db: Session, milestone_id: str, uploader_id: str) -> Tuple[Milestone, Project]:
        """Validates that the milestone and project exist, and uploader has proper NGO/Admin authority."""
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException(f"Milestone with identifier '{milestone_id}' not found.")

        project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none()
        if not project:
            raise NotFoundException(f"Project for milestone '{milestone_id}' not found.")

        user = db.execute(select(User).where(User.id == uploader_id)).scalar_one_or_none()
        if not user or not user.is_active:
            raise PermissionDeniedError("Active user account is required to upload evidence.")

        user_role_str = str(user.role.value if hasattr(user.role, "value") else user.role)
        if user_role_str in ("PLATFORM_ADMIN", "ADMIN"):
            return milestone, project

        # Check membership in the organization owning the project
        membership = db.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == project.organization_id,
                OrganizationMembership.user_id == uploader_id,
                OrganizationMembership.is_active == True
            )
        ).scalar_one_or_none()

        if not membership or membership.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER):
            raise PermissionDeniedError("User does not have authorized NGO membership to upload evidence for this project.")

        return milestone, project

    @staticmethod
    def _build_evidence_response(evidence: EvidenceFile, download_url: Optional[str] = None) -> EvidenceFileResponse:
        """Converts an EvidenceFile ORM entity to a Pydantic schema."""
        return EvidenceFileResponse(
            id=evidence.id,
            milestone_id=evidence.milestone_id,
            uploader_id=evidence.uploader_id,
            original_filename=evidence.original_filename,
            storage_key=evidence.storage_key,
            storage_provider=evidence.storage_provider,
            file_path=evidence.file_path,
            file_hash=evidence.file_hash,
            content_address=evidence.content_address,
            mime_type=evidence.mime_type,
            file_size_bytes=evidence.file_size_bytes,
            version_number=evidence.version_number,
            parent_evidence_id=evidence.parent_evidence_id,
            status=evidence.status.value if hasattr(evidence.status, "value") else str(evidence.status),
            is_archived=evidence.is_archived,
            archived_at=evidence.archived_at,
            quarantine_reason=evidence.quarantine_reason,
            created_at=evidence.created_at,
            download_url=download_url or evidence.file_path
        )

    @classmethod
    def upload_evidence_files(
        cls,
        db: Session,
        milestone_id: str,
        files: List[Tuple[bytes, str, str]],
        uploader_id: str
    ) -> EvidenceBatchUploadResponse:
        """
        Processes multi-file upload for a milestone.
        Enforces batch size limits, magic bytes validation, duplicate hash detection,
        secure storage persistence, and tamper-evident genesis audit log entry.
        """
        if not files:
            raise ValidationException("At least one evidence file must be provided for upload.")

        if len(files) > settings.MAX_UPLOAD_BATCH_SIZE:
            raise ValidationException(
                f"Batch size ({len(files)} files) exceeds maximum allowed {settings.MAX_UPLOAD_BATCH_SIZE} files."
            )

        milestone, project = cls._check_upload_permissions(db, milestone_id, uploader_id)
        storage = get_storage_provider()
        uploaded_records: List[EvidenceFile] = []

        for file_bytes, filename, mime_type in files:
            if not file_bytes:
                raise ValidationException(f"Evidence file '{filename}' cannot be empty.")

            max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
            file_size = len(file_bytes)
            if file_size > max_bytes:
                raise ValidationException(
                    f"File '{filename}' ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed {settings.MAX_UPLOAD_SIZE_MB} MB."
                )

            if mime_type not in settings.ALLOWED_MIME_TYPES:
                raise ValidationException(
                    f"File '{filename}' has unsupported MIME type '{mime_type}'. Allowed: {settings.ALLOWED_MIME_TYPES}"
                )

            if not _validate_magic_bytes(file_bytes, mime_type):
                raise ValidationException(
                    f"File '{filename}' contents do not match declared MIME type '{mime_type}' magic signature."
                )

            # Compute SHA-256
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            # Deduplication: Check if active file on same milestone has matching hash
            duplicate = db.execute(
                select(EvidenceFile).where(
                    EvidenceFile.milestone_id == milestone_id,
                    EvidenceFile.file_hash == file_hash,
                    EvidenceFile.is_archived == False
                )
            ).scalar_one_or_none()

            if duplicate:
                raise ConflictException(
                    f"Duplicate evidence detected: File with identical SHA-256 hash ({file_hash[:16]}...) already exists on milestone '{milestone_id}'."
                )

            # Persist to storage provider
            subfolder = f"milestones/{milestone_id}"
            storage_key, access_url, _, _ = storage.upload_file(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                subfolder=subfolder
            )

            # Create EvidenceFile record
            evidence = EvidenceFile(
                milestone_id=milestone_id,
                uploader_id=uploader_id,
                file_path=access_url,
                file_hash=file_hash,
                content_address=f"sha256:{file_hash}",
                mime_type=mime_type,
                detected_mime_type=mime_type,
                file_size_bytes=file_size,
                status=EvidenceStatus.UPLOADED,
                original_filename=filename,
                storage_key=storage_key,
                storage_provider=settings.STORAGE_PROVIDER,
                version_number=1,
                parent_evidence_id=None,
                is_archived=False
            )
            db.add(evidence)
            db.flush()

            # Genesis audit event for the tamper-evident hash chain
            genesis_prev_hash = "0" * 64
            now_iso = datetime.utcnow().isoformat()
            details = {
                "action": "UPLOAD",
                "original_filename": filename,
                "file_size_bytes": file_size,
                "mime_type": mime_type,
                "storage_key": storage_key,
                "storage_provider": settings.STORAGE_PROVIDER,
                "milestone_id": milestone_id,
                "timestamp": now_iso
            }
            curr_hash = compute_audit_hash(
                previous_hash=genesis_prev_hash,
                evidence_id=evidence.id,
                actor_id=uploader_id,
                event_type="UPLOADED",
                details=details
            )
            audit = EvidenceAuditEvent(
                evidence_id=evidence.id,
                actor_id=uploader_id,
                event_type="UPLOADED",
                details_json=details,
                previous_event_hash=genesis_prev_hash,
                current_event_hash=curr_hash
            )
            db.add(audit)
            uploaded_records.append(evidence)

        db.commit()
        for rec in uploaded_records:
            db.refresh(rec)

        return EvidenceBatchUploadResponse(
            milestone_id=milestone_id,
            total_uploaded=len(uploaded_records),
            items=[cls._build_evidence_response(rec) for rec in uploaded_records]
        )

    @classmethod
    def upload_evidence_version(
        cls,
        db: Session,
        parent_evidence_id: str,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        uploader_id: str,
        reason: Optional[str] = None
    ) -> EvidenceFileResponse:
        """
        Uploads a new version of an existing evidence asset.
        Preserves the historical version in place and links the new record.
        """
        parent = db.execute(select(EvidenceFile).where(EvidenceFile.id == parent_evidence_id)).scalar_one_or_none()
        if not parent or parent.is_archived:
            raise NotFoundException(f"Parent evidence asset '{parent_evidence_id}' not found or is archived.")

        cls._check_upload_permissions(db, parent.milestone_id, uploader_id)

        if not file_bytes:
            raise ValidationException("Replacement evidence file cannot be empty.")

        max_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
        file_size = len(file_bytes)
        if file_size > max_bytes:
            raise ValidationException(f"File exceeds maximum allowed {settings.MAX_UPLOAD_SIZE_MB} MB.")

        if mime_type not in settings.ALLOWED_MIME_TYPES:
            raise ValidationException(f"Unsupported MIME type '{mime_type}'.")

        if not _validate_magic_bytes(file_bytes, mime_type):
            raise ValidationException("File contents do not match MIME type magic signature.")

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Check duplicate hash against current active versions on milestone
        dup = db.execute(
            select(EvidenceFile).where(
                EvidenceFile.milestone_id == parent.milestone_id,
                EvidenceFile.file_hash == file_hash,
                EvidenceFile.is_archived == False
            )
        ).scalar_one_or_none()
        if dup:
            raise ConflictException("An active evidence file with identical SHA-256 hash already exists on this milestone.")

        # Determine next version number in chain
        chain_versions = db.execute(
            select(EvidenceFile.version_number).where(
                (EvidenceFile.id == parent.id) |
                (EvidenceFile.parent_evidence_id == parent.id) |
                (EvidenceFile.id == parent.parent_evidence_id)
            )
        ).scalars().all()
        next_version = (max(chain_versions) if chain_versions else parent.version_number) + 1

        storage = get_storage_provider()
        subfolder = f"milestones/{parent.milestone_id}"
        storage_key, access_url, _, _ = storage.upload_file(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            subfolder=subfolder
        )

        new_evidence = EvidenceFile(
            milestone_id=parent.milestone_id,
            uploader_id=uploader_id,
            file_path=access_url,
            file_hash=file_hash,
            content_address=f"sha256:{file_hash}",
            mime_type=mime_type,
            detected_mime_type=mime_type,
            file_size_bytes=file_size,
            status=EvidenceStatus.UPLOADED,
            original_filename=filename,
            storage_key=storage_key,
            storage_provider=settings.STORAGE_PROVIDER,
            version_number=next_version,
            parent_evidence_id=parent.id,
            is_archived=False
        )
        db.add(new_evidence)
        db.flush()

        # Cryptographic audit log linking to parent's latest event hash
        parent_latest_audit = db.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == parent.id).order_by(EvidenceAuditEvent.created_at.desc())
        ).scalars().first()
        prev_hash = parent_latest_audit.current_event_hash if parent_latest_audit else ("0" * 64)
        now_iso = datetime.utcnow().isoformat()

        details = {
            "action": "VERSION_CREATED",
            "parent_evidence_id": parent.id,
            "version_number": next_version,
            "reason": reason or "Evidence replacement",
            "original_filename": filename,
            "file_size_bytes": file_size,
            "file_hash": file_hash,
            "timestamp": now_iso
        }
        curr_hash = compute_audit_hash(
            previous_hash=prev_hash,
            evidence_id=new_evidence.id,
            actor_id=uploader_id,
            event_type="VERSION_CREATED",
            details=details
        )

        audit = EvidenceAuditEvent(
            evidence_id=new_evidence.id,
            actor_id=uploader_id,
            event_type="VERSION_CREATED",
            details_json=details,
            previous_event_hash=prev_hash,
            current_event_hash=curr_hash
        )
        db.add(audit)

        # Log SUPERSEDED on parent's audit log chain
        parent_details = {
            "action": "SUPERSEDED_BY_NEW_VERSION",
            "new_evidence_id": new_evidence.id,
            "new_version_number": next_version,
            "timestamp": now_iso
        }
        parent_curr_hash = compute_audit_hash(
            previous_hash=prev_hash,
            evidence_id=parent.id,
            actor_id=uploader_id,
            event_type="SUPERSEDED",
            details=parent_details
        )
        parent_audit = EvidenceAuditEvent(
            evidence_id=parent.id,
            actor_id=uploader_id,
            event_type="SUPERSEDED",
            details_json=parent_details,
            previous_event_hash=prev_hash,
            current_event_hash=parent_curr_hash
        )
        db.add(parent_audit)

        db.commit()
        db.refresh(new_evidence)
        return cls._build_evidence_response(new_evidence)

    @classmethod
    def get_evidence_download(
        cls,
        db: Session,
        evidence_id: str,
        user_id: Optional[str] = None
    ) -> Tuple[str, Optional[bytes], str, str]:
        """
        Validates authorization, logs a DOWNLOADED audit event, and returns
        (download_url, optional_bytes, filename, mime_type).
        """
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        # Access check
        if user_id:
            user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
            if not user or not user.is_active:
                raise PermissionDeniedError("Unauthorized access to evidence asset.")
            # Reviewers, Admins, or NGO members are allowed
            user_role_str = str(user.role.value if hasattr(user.role, "value") else user.role)
            if user_role_str not in ("PLATFORM_ADMIN", "ADMIN", "REVIEWER"):
                # Check org membership
                milestone = db.execute(select(Milestone).where(Milestone.id == evidence.milestone_id)).scalar_one_or_none()
                project = db.execute(select(Project).where(Project.id == milestone.project_id)).scalar_one_or_none()
                membership = db.execute(
                    select(OrganizationMembership).where(
                        OrganizationMembership.organization_id == project.organization_id,
                        OrganizationMembership.user_id == user_id,
                        OrganizationMembership.is_active == True
                    )
                ).scalar_one_or_none()
                if not membership:
                    raise PermissionDeniedError("Access to raw evidence asset is restricted to organization staff, reviewers, or administrators.")

        # Log DOWNLOADED audit event into the hash chain
        last_event = db.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == evidence.id).order_by(EvidenceAuditEvent.created_at.desc())
        ).scalars().first()
        prev_hash = last_event.current_event_hash if last_event else ("0" * 64)
        now_iso = datetime.utcnow().isoformat()
        details = {"action": "DOWNLOAD", "downloader_id": user_id, "timestamp": now_iso}
        curr_hash = compute_audit_hash(
            previous_hash=prev_hash,
            evidence_id=evidence.id,
            actor_id=user_id,
            event_type="DOWNLOADED",
            details=details
        )
        audit = EvidenceAuditEvent(
            evidence_id=evidence.id,
            actor_id=user_id,
            event_type="DOWNLOADED",
            details_json=details,
            previous_event_hash=prev_hash,
            current_event_hash=curr_hash
        )
        db.add(audit)
        db.commit()

        storage = get_storage_provider()
        filename = evidence.original_filename or f"evidence_{evidence.id[:8]}.dat"
        mime_type = evidence.mime_type or "application/octet-stream"

        if evidence.storage_provider in ("s3", "minio"):
            presigned_url = storage.generate_access_url(
                evidence.storage_key,
                expires_in_seconds=settings.PRESIGNED_URL_EXPIRATION_SECONDS
            )
            return presigned_url, None, filename, mime_type
        else:
            file_bytes = storage.get_file(evidence.storage_key)
            return evidence.file_path, file_bytes, filename, mime_type

    @classmethod
    def verify_evidence_integrity(
        cls,
        db: Session,
        evidence_id: str,
        actor_id: Optional[str] = None
    ) -> EvidenceIntegrityResponse:
        """
        Retrieves stored bytes, recalculates SHA-256 hash, and verifies cryptographic match.
        Logs an INTEGRITY_VERIFIED or INTEGRITY_TAMPERED audit event to the hash chain.
        """
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        storage = get_storage_provider()
        content = storage.get_file(evidence.storage_key)
        computed_sha256 = hashlib.sha256(content).hexdigest()
        is_valid = (computed_sha256.lower() == evidence.file_hash.lower())

        event_type = "INTEGRITY_VERIFIED" if is_valid else "INTEGRITY_TAMPERED"
        last_event = db.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == evidence.id).order_by(EvidenceAuditEvent.created_at.desc())
        ).scalars().first()
        prev_hash = last_event.current_event_hash if last_event else ("0" * 64)
        now_iso = datetime.utcnow().isoformat()
        details = {
            "expected_hash": evidence.file_hash,
            "computed_hash": computed_sha256,
            "is_valid": is_valid,
            "timestamp": now_iso
        }
        curr_hash = compute_audit_hash(
            previous_hash=prev_hash,
            evidence_id=evidence.id,
            actor_id=actor_id,
            event_type=event_type,
            details=details
        )
        audit = EvidenceAuditEvent(
            evidence_id=evidence.id,
            actor_id=actor_id,
            event_type=event_type,
            details_json=details,
            previous_event_hash=prev_hash,
            current_event_hash=curr_hash
        )
        db.add(audit)
        db.commit()

        return EvidenceIntegrityResponse(
            evidence_id=evidence.id,
            expected_hash=evidence.file_hash,
            computed_hash=computed_sha256,
            is_valid=is_valid,
            checked_at=datetime.utcnow(),
            storage_provider=evidence.storage_provider
        )

    @classmethod
    def archive_evidence(
        cls,
        db: Session,
        evidence_id: str,
        user_id: str,
        reason: Optional[str] = None
    ) -> EvidenceFileResponse:
        """
        Soft-archives an evidence file without destroying file bytes or historical records.
        """
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        if evidence.is_archived:
            return cls._build_evidence_response(evidence)

        # Check permissions
        cls._check_upload_permissions(db, evidence.milestone_id, user_id)

        evidence.is_archived = True
        evidence.archived_at = datetime.utcnow()

        last_event = db.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == evidence.id).order_by(EvidenceAuditEvent.created_at.desc())
        ).scalars().first()
        prev_hash = last_event.current_event_hash if last_event else ("0" * 64)
        now_iso = datetime.utcnow().isoformat()
        details = {
            "action": "ARCHIVE",
            "reason": reason or "Soft-archived by authorized user",
            "timestamp": now_iso
        }
        curr_hash = compute_audit_hash(
            previous_hash=prev_hash,
            evidence_id=evidence.id,
            actor_id=user_id,
            event_type="ARCHIVED",
            details=details
        )
        audit = EvidenceAuditEvent(
            evidence_id=evidence.id,
            actor_id=user_id,
            event_type="ARCHIVED",
            details_json=details,
            previous_event_hash=prev_hash,
            current_event_hash=curr_hash
        )
        db.add(audit)
        db.commit()
        db.refresh(evidence)
        return cls._build_evidence_response(evidence)

    @classmethod
    def list_milestone_evidence(
        cls,
        db: Session,
        milestone_id: str,
        include_archived: bool = False
    ) -> List[EvidenceFileResponse]:
        """Lists evidence assets for a given milestone."""
        query = select(EvidenceFile).where(EvidenceFile.milestone_id == milestone_id)
        if not include_archived:
            query = query.where(EvidenceFile.is_archived == False)
        query = query.order_by(EvidenceFile.created_at.desc())

        items = db.execute(query).scalars().all()
        return [cls._build_evidence_response(item) for item in items]

    @classmethod
    def get_evidence_detail(cls, db: Session, evidence_id: str) -> EvidenceFileResponse:
        """Retrieves single evidence asset by ID."""
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")
        return cls._build_evidence_response(evidence)

    @classmethod
    def get_version_history(cls, db: Session, evidence_id: str) -> EvidenceVersionHistoryResponse:
        """Retrieves all versions in the lineage of this evidence asset."""
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        # Find root parent
        root = evidence
        while root.parent_evidence_id:
            parent = db.execute(select(EvidenceFile).where(EvidenceFile.id == root.parent_evidence_id)).scalar_one_or_none()
            if not parent:
                break
            root = parent

        # Query all descendants starting from root
        all_versions = db.execute(
            select(EvidenceFile).where(
                (EvidenceFile.id == root.id) |
                (EvidenceFile.parent_evidence_id == root.id)
            ).order_by(EvidenceFile.version_number.asc())
        ).scalars().all()

        return EvidenceVersionHistoryResponse(
            evidence_id=evidence.id,
            current_version=evidence.version_number,
            versions=[cls._build_evidence_response(v) for v in all_versions]
        )

    @classmethod
    def get_audit_trail(cls, db: Session, evidence_id: str) -> EvidenceAuditLogResponse:
        """Retrieves and cryptographically verifies the SHA-256 audit hash chain for an evidence asset."""
        evidence = db.execute(select(EvidenceFile).where(EvidenceFile.id == evidence_id)).scalar_one_or_none()
        if not evidence:
            raise NotFoundException(f"Evidence file '{evidence_id}' not found.")

        events = db.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == evidence.id).order_by(EvidenceAuditEvent.created_at.asc())
        ).scalars().all()

        # Cryptographically verify the hash chain
        is_chain_intact = True
        expected_prev_hash = None

        for idx, ev in enumerate(events):
            if idx > 0 and ev.previous_event_hash != expected_prev_hash:
                is_chain_intact = False
                break
            computed = compute_audit_hash(
                previous_hash=ev.previous_event_hash,
                evidence_id=ev.evidence_id,
                actor_id=ev.actor_id,
                event_type=ev.event_type,
                details=ev.details_json
            )
            if computed != ev.current_event_hash:
                is_chain_intact = False
                break
            expected_prev_hash = ev.current_event_hash

        event_schemas = [
            EvidenceAuditEventResponse(
                id=ev.id,
                evidence_id=ev.evidence_id,
                actor_id=ev.actor_id,
                event_type=ev.event_type,
                details_json=ev.details_json,
                previous_event_hash=ev.previous_event_hash,
                current_event_hash=ev.current_event_hash,
                created_at=ev.created_at
            )
            for ev in events
        ]

        return EvidenceAuditLogResponse(
            evidence_id=evidence.id,
            is_chain_intact=is_chain_intact,
            total_events=len(events),
            events=event_schemas
        )
