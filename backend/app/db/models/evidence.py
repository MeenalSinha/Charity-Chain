"""
evidence.py - Evidence Files, Forensic Reports & Verification Check Records
===========================================================================
Defines models for milestone evidence storage, multi-versioning,
tamper-evident SHA-256 audit log chains, and verification records.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey,
    Text, DateTime, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship

from backend.app.db.base import Base
from backend.app.core.constants import EvidenceStatus, VerificationCheckStatus


def compute_audit_hash(
    previous_hash: str,
    evidence_id: str,
    actor_id: Optional[str],
    event_type: str,
    details: Any,
    timestamp_iso: Optional[str] = None
) -> str:
    """
    Computes a deterministic, tamper-evident SHA-256 hash for an audit log entry.
    Chains cryptographically to the previous event hash.
    """
    canonical_details = json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
    if timestamp_iso is not None:
        payload = f"{previous_hash}|{evidence_id}|{actor_id or ''}|{event_type}|{canonical_details}|{timestamp_iso}"
    else:
        payload = f"{previous_hash}|{evidence_id}|{actor_id or ''}|{event_type}|{canonical_details}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvidenceFile(Base):
    """
    Stores immutable or versioned evidence assets uploaded by NGOs for milestone completion.
    Linked to local or cloud object storage with SHA-256 checksums and version history.
    """
    __tablename__ = "evidence_files"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    uploader_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    content_address = Column(String(128), nullable=False)
    mime_type = Column(String(64), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    status = Column(SQLEnum(EvidenceStatus), default=EvidenceStatus.UPLOADED, nullable=False)

    # Phase 5 Storage & Versioning Extensions
    original_filename = Column(String(255), nullable=True)
    storage_key = Column(String(500), nullable=True)
    storage_provider = Column(String(50), nullable=False, default="local")
    storage_version_id = Column(String(100), nullable=True)
    detected_mime_type = Column(String(100), nullable=True)
    version_number = Column(Integer, nullable=False, default=1)
    parent_evidence_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="SET NULL"), nullable=True, index=True)
    is_archived = Column(Boolean, nullable=False, default=False, index=True)
    archived_at = Column(DateTime, nullable=True)
    quarantine_reason = Column(String(255), nullable=True)

    # Relationships
    milestone = relationship("Milestone", back_populates="evidence_files")
    uploader = relationship("User", foreign_keys=[uploader_id])
    parent_evidence = relationship("EvidenceFile", remote_side="EvidenceFile.id", foreign_keys=[parent_evidence_id], backref="child_versions")
    verification_reports = relationship("VerificationReport", back_populates="evidence_file", cascade="all, delete-orphan")
    audit_events = relationship("EvidenceAuditEvent", back_populates="evidence_file", cascade="all, delete-orphan", order_by="EvidenceAuditEvent.created_at")


class EvidenceAuditEvent(Base):
    """
    Tamper-evident audit ledger entry for evidence lifecycle events.
    Forms a cryptographic SHA-256 hash chain guaranteeing immutable event order.
    """
    __tablename__ = "evidence_audit_events"

    evidence_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)  # UPLOADED, INTEGRITY_VERIFIED, VERSION_CREATED, ARCHIVED, DOWNLOADED
    details_json = Column(JSON, nullable=True)
    previous_event_hash = Column(String(64), nullable=False)
    current_event_hash = Column(String(64), nullable=False, index=True)

    evidence_file = relationship("EvidenceFile", back_populates="audit_events")
    actor = relationship("User", foreign_keys=[actor_id])


class VerificationReport(Base):
    """
    Stores comprehensive, multi-indicator AI and digital forensics verification reports.
    Immutable: historical reports are never overwritten when new reports are generated.
    """
    __tablename__ = "verification_reports"

    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_version = Column(Integer, nullable=False, default=1)
    evidence_hash = Column(String(64), nullable=False, default="")
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=True, index=True)
    pipeline_version = Column(String(50), nullable=False, default="v6.0.0")
    overall_assessment = Column(String(50), nullable=False, default="NEEDS_FURTHER_REVIEW", index=True)
    confidence_score = Column(Float, nullable=True)
    calibration_status = Column(String(50), nullable=False, default="UNVALIDATED")
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    provider_name = Column(String(100), nullable=False, default="yolo_forensics")
    model_metadata_json = Column(JSON, nullable=True)
    limitations_json = Column(JSON, nullable=True)
    config_snapshot_json = Column(JSON, nullable=True)
    error_details = Column(Text, nullable=True)

    # Legacy compatibility fields
    overall_confidence = Column(Float, nullable=False, default=0.0)
    overall_pass = Column(Boolean, nullable=False, default=False)
    reviewed_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    summary_notes = Column(Text, nullable=True)

    # Relationships
    evidence_file = relationship("EvidenceFile", back_populates="verification_reports")
    checks = relationship("VerificationCheckRecord", back_populates="report", cascade="all, delete-orphan", order_by="VerificationCheckRecord.created_at")


class VerificationCheckRecord(Base):
    """
    Individual technical indicator check record within a VerificationReport.
    Captures raw measurements, algorithm parameters, and diagnostic artifacts.
    """
    __tablename__ = "verification_check_records"

    report_id = Column(String(36), ForeignKey("verification_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    check_type = Column(String(50), nullable=False, index=True)  # YOLO_OBJECTS, ELA_TAMPER, PHASH_SIMILARITY, SSIM_SIMILARITY, etc.
    status = Column(SQLEnum(VerificationCheckStatus), nullable=False)
    passed = Column(Boolean, nullable=False, default=False)
    message = Column(Text, nullable=False)
    technical_result_json = Column(JSON, nullable=True)
    raw_metrics_json = Column(JSON, nullable=True)
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(50), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    diagnostic_artifact_path = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    details_json = Column(Text, nullable=True)

    report = relationship("VerificationReport", back_populates="checks")


class VerificationJob(Base):
    """
    Durable background verification job tracking for asynchronous execution,
    idempotency protection, and fault-tolerant retry handling.
    """
    __tablename__ = "verification_jobs"

    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_hash = Column(String(64), nullable=False)
    evidence_version = Column(Integer, nullable=False, default=1)
    requested_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(50), nullable=False, default="PENDING", index=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    pipeline_version = Column(String(50), nullable=False, default="v6.0.0")
    requested_checks_json = Column(JSON, nullable=True)
    baseline_evidence_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="SET NULL"), nullable=True)
    report_id = Column(String(36), ForeignKey("verification_reports.id", ondelete="SET NULL"), nullable=True)
    idempotency_key = Column(String(128), nullable=True, unique=True, index=True)
    error_category = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    queued_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    evidence_file = relationship("EvidenceFile", foreign_keys=[evidence_file_id])
    baseline_evidence = relationship("EvidenceFile", foreign_keys=[baseline_evidence_id])
    report = relationship("VerificationReport", foreign_keys=[report_id])
    requested_by = relationship("User", foreign_keys=[requested_by_id])
