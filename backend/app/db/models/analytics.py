"""
analytics.py - Database Models for Analytics, Snapshots, and Public Transparency
================================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Text, Boolean, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from backend.app.db.base import Base


class AnalyticsSnapshot(Base):
    """
    Persisted periodic or custom analytical aggregations.
    Allows point-in-time reproducibility and cached summary retrieval
    without altering or overwriting underlying authoritative transactional records.
    """
    __tablename__ = "analytics_snapshots"

    scope = Column(String(50), nullable=False, default="PLATFORM", index=True)  # PLATFORM, ORGANIZATION, DONOR, PUBLIC
    scope_id = Column(String(36), nullable=True, index=True)  # organization_id, donor_id, or NULL for platform/public
    snapshot_type = Column(String(50), nullable=False, default="OVERVIEW", index=True)  # OVERVIEW, FUNDING, MILESTONES, GEOGRAPHY
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    as_of_date = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    currency = Column(String(3), nullable=False, default="USD")
    metrics_json = Column(Text, nullable=False)  # Serialized dictionary of computed metrics
    is_valid = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_analytics_snapshots_scope_type", "scope", "snapshot_type", "is_valid"),
    )


class AnalyticsExportJob(Base):
    """
    Tracks requested analytical report exports (CSV, PDF)
    with role-based access control, file tracking, and asynchronous status tracking.
    """
    __tablename__ = "analytics_export_jobs"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True)
    export_type = Column(String(50), nullable=False, index=True)  # NGO_IMPACT, DONOR_PORTFOLIO, ADMIN_FINANCIAL, PUBLIC_TRANSPARENCY
    export_format = Column(String(10), nullable=False, default="CSV")  # CSV, PDF
    status = Column(String(50), nullable=False, default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    file_path = Column(String(512), nullable=True)
    download_url = Column(String(512), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    parameters_json = Column(Text, nullable=True)  # Date ranges, filters, options
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", foreign_keys=[user_id])
    organization = relationship("Organization", foreign_keys=[organization_id])


class PublicTransparencyPublication(Base):
    """
    Explicitly published public transparency records.
    Controls what project or organization information is approved for public indexing
    and transparency portal presentation.
    """
    __tablename__ = "public_transparency_publications"

    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    published_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    publication_type = Column(String(50), nullable=False, default="PROJECT_SUMMARY", index=True)
    is_published = Column(Boolean, nullable=False, default=True, index=True)
    published_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    methodology_version = Column(String(50), nullable=False, default="v1.0")
    summary_data_json = Column(Text, nullable=True)

    project = relationship("Project", foreign_keys=[project_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    published_by = relationship("User", foreign_keys=[published_by_id])

