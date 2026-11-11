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
    ReconciliationResult, DisputeStatus, DisputeCategory, DisputeResolution
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
    __tablename__ = "impact_certificates"

    donation_id = Column(String(36), ForeignKey("donations.id", ondelete="CASCADE"), unique=True, nullable=False)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False)
    certificate_number = Column(String(64), unique=True, nullable=False, index=True)
    recipient_name = Column(String(255), nullable=False)
    certificate_hash = Column(String(64), nullable=False, index=True)
    token_uri = Column(String(512), nullable=False)
    metadata_json = Column(Text, nullable=True)

    donation = relationship("Donation", back_populates="certificate")


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
