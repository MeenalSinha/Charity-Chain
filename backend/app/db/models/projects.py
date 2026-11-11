"""
projects.py - Project, Category, Milestone & Geolocation Models
===============================================================
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, ForeignKey, Text, Boolean,
    DateTime, UniqueConstraint, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import ProjectStatus, MilestoneStatus, ProjectVisibility


class ProjectCategory(Base):
    __tablename__ = "project_categories"

    name = Column(String(100), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    projects = relationship("Project", back_populates="category")


class Project(Base):
    __tablename__ = "projects"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    category_id = Column(String(36), ForeignKey("project_categories.id"), nullable=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    summary = Column(String(500), nullable=True)
    description = Column(Text, nullable=False)
    objectives = Column(Text, nullable=True)
    project_type = Column(String(50), default="CHARITABLE_PROJECT", nullable=False)
    public_visibility = Column(SQLEnum(ProjectVisibility), default=ProjectVisibility.PUBLIC, nullable=False)
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.DRAFT, nullable=False, index=True)
    
    # Financial Targets
    target_amount_minor_units = Column(Integer, nullable=False, default=0)
    raised_amount_minor_units = Column(Integer, nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    
    # Impact & Deliverables
    target_beneficiaries = Column(String(255), nullable=True)
    expected_outcomes = Column(Text, nullable=True)
    deliverables = Column(Text, nullable=True)
    impact_metrics = Column(Text, nullable=True)

    # Timeline
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)

    # Geographic Location
    country = Column(String(3), nullable=True, default="USA")
    state_region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    address = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    radius_meters = Column(Float, nullable=False, default=1000.0)

    # Lifecycle & Audit Tracking
    creator_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    revision_notes = Column(Text, nullable=True)
    version_number = Column(Integer, default=1, nullable=False)
    is_archived = Column(Boolean, default=False, nullable=False)

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    category = relationship("ProjectCategory", back_populates="projects")
    milestones = relationship("Milestone", back_populates="project", cascade="all, delete-orphan", order_by="Milestone.sequence_number")
    creator = relationship("User", foreign_keys=[creator_id])
    reviewer = relationship("User", foreign_keys=[reviewer_id])


class Milestone(Base):
    __tablename__ = "milestones"
    __table_args__ = (
        UniqueConstraint("project_id", "sequence_number", name="uq_milestone_project_seq"),
    )

    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    objective = Column(Text, nullable=True)
    deliverables = Column(Text, nullable=True)
    expected_outcome = Column(Text, nullable=True)
    success_criteria = Column(Text, nullable=True)
    required_evidence_desc = Column(Text, nullable=True)
    
    sequence_number = Column(Integer, nullable=False, default=1)
    target_amount_minor_units = Column(Integer, nullable=False, default=0)
    raised_amount_minor_units = Column(Integer, nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(MilestoneStatus), default=MilestoneStatus.DRAFT, nullable=False, index=True)

    planned_start_date = Column(DateTime, nullable=True)
    planned_completion_date = Column(DateTime, nullable=True)

    # Geolocation override
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    radius_meters = Column(Float, nullable=False, default=1000.0)
    address = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    country = Column(String(3), nullable=True)
    
    version_number = Column(Integer, default=1, nullable=False)

    project = relationship("Project", back_populates="milestones")
    location = relationship("ProjectLocation", back_populates="milestone", uselist=False, cascade="all, delete-orphan")
    donations = relationship("Donation", back_populates="milestone")
    evidence_files = relationship("EvidenceFile", back_populates="milestone")


class ProjectLocation(Base):
    __tablename__ = "project_locations"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), unique=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    country = Column(String(3), nullable=True)
    radius_meters = Column(Float, nullable=False, default=1000.0)

    milestone = relationship("Milestone", back_populates="location")

