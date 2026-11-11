"""
reviews_impact.py - Human Reviews, Proof-of-Impact Certificates & Audit Log
===========================================================================
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import ReviewDecision


class ReviewTask(Base):
    __tablename__ = "review_tasks"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False)
    evidence_file_id = Column(String(36), ForeignKey("evidence_files.id", ondelete="CASCADE"), nullable=False)
    assigned_reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(50), default="PENDING", nullable=False)

    decisions = relationship("ReviewDecisionRecord", back_populates="task", cascade="all, delete-orphan")


class ReviewDecisionRecord(Base):
    __tablename__ = "review_decision_records"

    task_id = Column(String(36), ForeignKey("review_tasks.id", ondelete="CASCADE"), nullable=False)
    reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    decision = Column(SQLEnum(ReviewDecision), nullable=False)
    comments = Column(Text, nullable=False)
    decided_at = Column(DateTime, nullable=False)

    task = relationship("ReviewTask", back_populates="decisions")


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

