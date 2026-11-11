"""
projects.py - Project, Category, Milestone & Geolocation Models
===============================================================
"""

from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import ProjectStatus, MilestoneStatus


class ProjectCategory(Base):
    __tablename__ = "project_categories"

    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    projects = relationship("Project", back_populates="category")


class Project(Base):
    __tablename__ = "projects"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    category_id = Column(String(36), ForeignKey("project_categories.id"), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.DRAFT, nullable=False)
    target_amount_minor_units = Column(Integer, nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")

    organization = relationship("Organization", back_populates="projects")
    category = relationship("ProjectCategory", back_populates="projects")
    milestones = relationship("Milestone", back_populates="project", cascade="all, delete-orphan", order_by="Milestone.sequence_number")


class Milestone(Base):
    __tablename__ = "milestones"

    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    sequence_number = Column(Integer, nullable=False, default=1)
    target_amount_minor_units = Column(Integer, nullable=False, default=0)
    raised_amount_minor_units = Column(Integer, nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(MilestoneStatus), default=MilestoneStatus.DRAFT, nullable=False)

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

