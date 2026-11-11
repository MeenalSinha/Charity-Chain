"""
evidence.py - Evidence Files, Forensic Reports & Verification Check Records
===========================================================================
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import EvidenceStatus, VerificationCheckStatus


class EvidenceFile(Base):
    __tablename__ = "evidence_files"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False)
    uploader_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    content_address = Column(String(128), nullable=False)
    mime_type = Column(String(64), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    status = Column(SQLEnum(EvidenceStatus), default=EvidenceStatus.UPLOADED, nullable=False)

    milestone = relationship("Milestone", back_populates="evidence_files")
    verification_reports = relationship("VerificationReport", back_populates="evidence_file", cascade="all, delete-orphan")


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

