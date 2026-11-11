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
    __tablename__ = "verification_reports"

    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False)
    overall_confidence = Column(Float, nullable=False)
    overall_pass = Column(Boolean, nullable=False)
    reviewed_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    summary_notes = Column(Text, nullable=True)

    evidence_file = relationship("EvidenceFile", back_populates="verification_reports")
    checks = relationship("VerificationCheckRecord", back_populates="report", cascade="all, delete-orphan")


class VerificationCheckRecord(Base):
    __tablename__ = "verification_check_records"

    report_id = Column(String(36), ForeignKey("verification_reports.id", ondelete="CASCADE"), nullable=False)
    check_type = Column(String(50), nullable=False)  # GPS, TAMPER, OBJECTS, BEFORE_AFTER
    status = Column(SQLEnum(VerificationCheckStatus), nullable=False)
    passed = Column(Boolean, nullable=False)
    message = Column(Text, nullable=False)
    details_json = Column(Text, nullable=True)

    report = relationship("VerificationReport", back_populates="checks")
