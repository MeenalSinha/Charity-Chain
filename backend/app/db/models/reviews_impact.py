"""
reviews_impact.py - Human Reviews, Dual-Key Approvals, Disputes & Audit Trail
=============================================================================
Enforces human decision-making governance, reviewer assignment, conflict-of-interest checks,
dual-key reconciliation, structured revision requests, dispute resolution, and tamper-evident audit history.
"""

import uuid
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum, JSON
)
from sqlalchemy.orm import relationship

from backend.app.db.base import Base
from backend.app.core.constants import (
    ReviewDecision, ReviewCaseStatus, ReviewAssignmentStatus,
    ReconciliationResult, DisputeStatus, DisputeCategory, DisputeResolution,
    CertificateType, CertificateStatus, CertificateRevocationReason,
    ImpactLedgerEventType, SigningKeyStatus
)


def compute_review_audit_hash(
    previous_hash: str,
    milestone_id: str,
    actor_id: Optional[str],
    event_type: str,
    details: Dict[str, Any]
) -> str:
    """
    Computes a deterministic SHA-256 hash linking each review audit event
    to its immediate predecessor, establishing an unbroken audit chain.
    """
    canonical_details = json.dumps(details, sort_keys=True, separators=(",", ":"))
    payload = f"{previous_hash}|{milestone_id}|{actor_id or ''}|{event_type}|{canonical_details}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReviewCase(Base):
    """
    Represents an atomic milestone submission round undergoing independent human review.
    """
    __tablename__ = "review_cases"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    submission_round = Column(Integer, default=1, nullable=False)
    submitted_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    status = Column(String(50), default=ReviewCaseStatus.SUBMITTED.value, nullable=False, index=True)
    final_decision = Column(String(50), nullable=True)
    reconciliation_result = Column(String(50), nullable=True)
    submission_notes = Column(Text, nullable=True)

    snapshot_digest = Column(String(64), nullable=False, index=True)
    snapshot_data_json = Column(JSON, nullable=True)
    review_policy_version = Column(String(32), default="v7.0.0", nullable=False)

    revision_deadline = Column(DateTime, nullable=True)
    appeal_deadline = Column(DateTime, nullable=True)

    adjudicator_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    adjudication_comments = Column(Text, nullable=True)
    adjudicated_at = Column(DateTime, nullable=True)
    finalized_at = Column(DateTime, nullable=True)

    # Relationships
    milestone = relationship("Milestone")
    project = relationship("Project")
    organization = relationship("Organization")
    submitted_by = relationship("User", foreign_keys=[submitted_by_id])
    adjudicator = relationship("User", foreign_keys=[adjudicator_id])

    tasks = relationship("ReviewTask", back_populates="review_case", cascade="all, delete-orphan")
    evidence_items = relationship("ReviewCaseEvidence", back_populates="review_case", cascade="all, delete-orphan")
    decisions = relationship("ReviewDecisionRecord", back_populates="review_case", cascade="all, delete-orphan")
    disputes = relationship("ReviewDispute", back_populates="review_case", cascade="all, delete-orphan")


class ReviewCaseEvidence(Base):
    """
    Binds a review case to the exact evidence file IDs, versions, hashes,
    and associated Phase 6 AI verification reports evaluated during that round.
    """
    __tablename__ = "review_case_evidence"

    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_version = Column(Integer, nullable=False)
    evidence_hash = Column(String(64), nullable=False)
    verification_report_id = Column(String(36), ForeignKey("verification_reports.id", ondelete="SET NULL"), nullable=True)
    ai_pipeline_version = Column(String(32), nullable=True)
    evidence_role = Column(String(64), default="PRIMARY", nullable=False)
    snapshot_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    review_case = relationship("ReviewCase", back_populates="evidence_items")
    evidence_file = relationship("EvidenceFile")
    verification_report = relationship("VerificationReport")


class ReviewTask(Base):
    """
    Represents an individual reviewer's assigned slot on a review case.
    Supports dual-key independent slots (Slot 1 and Slot 2).
    """
    __tablename__ = "review_tasks"

    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=True, index=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="SET NULL"), nullable=True)
    assigned_reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    reviewer_slot = Column(Integer, default=1, nullable=False)

    status = Column(String(50), default="ASSIGNED", nullable=False, index=True)
    assignment_state = Column(String(50), default="ASSIGNED", nullable=False)

    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    due_at = Column(DateTime, nullable=True)
    decided_at = Column(DateTime, nullable=True)

    decline_reason = Column(Text, nullable=True)
    conflict_declared = Column(Boolean, default=False, nullable=False)
    conflict_reason = Column(Text, nullable=True)
    reassigned_from_task_id = Column(String(36), nullable=True)

    # Relationships
    review_case = relationship("ReviewCase", back_populates="tasks")
    assigned_reviewer = relationship("User", foreign_keys=[assigned_reviewer_id])
    decisions = relationship("ReviewDecisionRecord", back_populates="task", cascade="all, delete-orphan")


class ReviewDecisionRecord(Base):
    """
    Stores an immutable, cryptographically snapshot-bound decision
    submitted by an assigned reviewer for their review task slot.
    """
    __tablename__ = "review_decision_records"

    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=True, index=True)
    task_id = Column(String(36), ForeignKey("review_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    decision = Column(SQLEnum(ReviewDecision), nullable=False)
    comments = Column(Text, nullable=False)

    structured_revision_data_json = Column(JSON, nullable=True)
    rejection_reason_category = Column(String(100), nullable=True)
    evidence_snapshot_digest = Column(String(64), nullable=True)
    ai_report_id = Column(String(36), nullable=True)
    submission_round = Column(Integer, default=1, nullable=False)
    is_superseded = Column(Boolean, default=False, nullable=False)

    decided_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    task = relationship("ReviewTask", back_populates="decisions")
    review_case = relationship("ReviewCase", back_populates="decisions")
    reviewer = relationship("User", foreign_keys=[reviewer_id])


class ReviewDispute(Base):
    """
    Represents a formal appeal or dispute submitted by an NGO against
    an adverse final milestone review decision (e.g., REJECTED).
    """
    __tablename__ = "review_disputes"

    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    original_decision_record_id = Column(String(36), nullable=True)
    raised_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    status = Column(String(50), default=DisputeStatus.OPEN.value, nullable=False, index=True)
    dispute_category = Column(String(100), nullable=False)
    dispute_reason = Column(Text, nullable=False)
    supporting_evidence_json = Column(JSON, nullable=True)

    assigned_adjudicator_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolution = Column(String(50), nullable=True)
    resolution_explanation = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    review_case = relationship("ReviewCase", back_populates="disputes")
    milestone = relationship("Milestone")
    project = relationship("Project")
    organization = relationship("Organization")
    raised_by = relationship("User", foreign_keys=[raised_by_id])
    assigned_adjudicator = relationship("User", foreign_keys=[assigned_adjudicator_id])
    messages = relationship("ReviewDisputeMessage", back_populates="dispute", cascade="all, delete-orphan")


class ReviewDisputeMessage(Base):
    """
    Auditable communication record within a formal dispute case thread.
    """
    __tablename__ = "review_dispute_messages"

    dispute_id = Column(String(36), ForeignKey("review_disputes.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    sender_role = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    attachments_json = Column(JSON, nullable=True)

    # Relationships
    dispute = relationship("ReviewDispute", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id])


class ReviewAuditEvent(Base):
    """
    Cryptographic SHA-256 hash-chained event record providing application-level
    tamper evidence for all reviewer assignments, decisions, revisions, and disputes.
    """
    __tablename__ = "review_audit_events"

    review_case_id = Column(String(36), nullable=True, index=True)
    milestone_id = Column(String(36), nullable=False, index=True)
    actor_id = Column(String(36), nullable=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    details_json = Column(JSON, nullable=True)
    previous_event_hash = Column(String(64), nullable=False)
    current_event_hash = Column(String(64), nullable=False, index=True)


class ImpactCertificate(Base):
    """
    Modernized cryptographically signed Proof-of-Impact Certificate.
    Protects against cascading deletion and contains versioned canonical claim payloads,
    Ed25519 asymmetric signatures, and public verification tokens.
    """
    __tablename__ = "impact_certificates"

    donation_id = Column(String(36), ForeignKey("donations.id", ondelete="RESTRICT"), nullable=False, index=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="RESTRICT"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="RESTRICT"), nullable=True, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=True, index=True)
    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True, index=True)
    disbursement_id = Column(String(36), ForeignKey("disbursements.id", ondelete="SET NULL"), nullable=True, index=True)

    certificate_number = Column(String(64), unique=True, nullable=False, index=True)
    verification_id = Column(String(64), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    idempotency_key = Column(String(128), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    certificate_type = Column(String(50), default=CertificateType.CONTRIBUTION_IMPACT.value, nullable=False, index=True)
    schema_version = Column(String(20), default="v9.0.0", nullable=False)

    recipient_name = Column(String(255), nullable=False, default="")
    public_attribution = Column(Boolean, default=False, nullable=False)
    public_display_name = Column(String(255), nullable=True)

    canonical_payload = Column(Text, nullable=False, default="")
    payload_hash = Column(String(64), nullable=False, index=True, default="")
    signature = Column(Text, nullable=False, default="")
    signature_algorithm = Column(String(32), default="Ed25519", nullable=False)
    signing_key_id = Column(String(64), nullable=False, index=True, default="key-charitychain-2026-v1")
    issuer_id = Column(String(128), default="charitychain-core-v1", nullable=False)

    status = Column(String(32), default=CertificateStatus.ISSUED.value, nullable=False, index=True)
    revocation_reason = Column(String(100), nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    superseded_by_id = Column(String(36), ForeignKey("impact_certificates.id", ondelete="SET NULL"), nullable=True)

    pdf_storage_key = Column(String(255), nullable=True)
    pdf_hash = Column(String(64), nullable=True)
    qr_code_url = Column(String(512), nullable=True)

    # Legacy fields preserved for backward compatibility
    certificate_hash = Column(String(64), nullable=False, index=True, default="")
    token_uri = Column(String(512), nullable=False, default="")
    metadata_json = Column(Text, nullable=True)

    donation = relationship("Donation", back_populates="certificates")
    milestone = relationship("Milestone")
    project = relationship("Project")
    organization = relationship("Organization")
    review_case = relationship("ReviewCase")
    disbursement = relationship("Disbursement")
    status_events = relationship("CertificateStatusEvent", back_populates="certificate", cascade="all, delete-orphan")
    superseded_by = relationship("ImpactCertificate", remote_side="ImpactCertificate.id")


class CertificateStatusEvent(Base):
    """
    Audit log of certificate lifecycle transitions (issued, suspended, revoked, superseded).
    """
    __tablename__ = "certificate_status_events"

    certificate_id = Column(String(36), ForeignKey("impact_certificates.id", ondelete="CASCADE"), nullable=False, index=True)
    previous_status = Column(String(32), nullable=False)
    new_status = Column(String(32), nullable=False)
    reason_code = Column(String(100), nullable=False)
    reason_description = Column(Text, nullable=True)
    actor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    certificate = relationship("ImpactCertificate", back_populates="status_events")


class ImpactLedgerStream(Base):
    """
    Represents an append-only event stream (e.g. per Milestone, Project, or Global).
    Tracks the atomic sequence counter and current cryptographic head hash.
    """
    __tablename__ = "impact_ledger_streams"

    stream_id = Column(String(128), unique=True, nullable=False, index=True)
    aggregate_type = Column(String(50), nullable=False, index=True)
    aggregate_id = Column(String(64), nullable=False, index=True)
    current_sequence_number = Column(Integer, default=0, nullable=False)
    head_event_hash = Column(String(64), default="0" * 64, nullable=False)
    status = Column(String(32), default="ACTIVE", nullable=False)

    events = relationship("ImpactLedgerEvent", back_populates="stream", order_by="ImpactLedgerEvent.sequence_number")


class ImpactLedgerEvent(Base):
    """
    Cryptographically chained, immutable impact event linking previous event hash to current event hash.
    """
    __tablename__ = "impact_ledger_events"

    stream_id = Column(String(128), ForeignKey("impact_ledger_streams.stream_id", ondelete="RESTRICT"), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)
    event_type = Column(String(64), nullable=False, index=True)
    schema_version = Column(String(20), default="v9.0.0", nullable=False)
    source_event_type = Column(String(64), nullable=False)
    source_event_id = Column(String(128), nullable=False)

    payload_json = Column(Text, nullable=False)
    payload_hash = Column(String(64), nullable=False, index=True)
    previous_event_hash = Column(String(64), nullable=False)
    current_event_hash = Column(String(64), nullable=False, index=True)

    actor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    stream = relationship("ImpactLedgerStream", back_populates="events")


class ImpactLedgerCheckpoint(Base):
    """
    Cryptographically signed checkpoint of an Impact Ledger stream head.
    """
    __tablename__ = "impact_ledger_checkpoints"

    stream_id = Column(String(128), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)
    head_hash = Column(String(64), nullable=False)
    signing_key_id = Column(String(64), nullable=False)
    signature_algorithm = Column(String(32), default="Ed25519", nullable=False)
    signature = Column(Text, nullable=False)
    checkpoint_metadata_json = Column(Text, nullable=True)


class IssuerSigningKey(Base):
    """
    Public key registry and lifecycle metadata for Ed25519 signing keys.
    Private key material is NEVER stored in database tables.
    """
    __tablename__ = "issuer_signing_keys"

    key_id = Column(String(64), unique=True, nullable=False, index=True)
    algorithm = Column(String(32), default="Ed25519", nullable=False)
    public_key_pem = Column(Text, nullable=False)
    public_key_jwk = Column(Text, nullable=False)
    status = Column(String(32), default=SigningKeyStatus.ACTIVE.value, nullable=False, index=True)
    usage = Column(String(32), default="SIG", nullable=False)

    activated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    retired_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)



class AuditEvent(Base):
    __tablename__ = "audit_events"

    actor_id = Column(String(36), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False, index=True)
    entity_id = Column(String(64), nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    correlation_id = Column(String(64), nullable=True, index=True)
    details_json = Column(Text, nullable=True)

    def __init__(self, **kwargs):
        import json
        if "entity_name" in kwargs and "entity_type" not in kwargs:
            kwargs["entity_type"] = kwargs.pop("entity_name")
        if "after_state" in kwargs and "details_json" not in kwargs:
            val = kwargs.pop("after_state")
            kwargs["details_json"] = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        super().__init__(**kwargs)

    @property
    def entity_name(self) -> str:
        return self.entity_type

    @entity_name.setter
    def entity_name(self, val: str):
        self.entity_type = val

    @property
    def after_state(self):
        import json
        if self.details_json:
            try:
                return json.loads(self.details_json)
            except Exception:
                return self.details_json
        return None
