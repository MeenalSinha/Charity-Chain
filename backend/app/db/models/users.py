"""
users.py - Identity, Organization & NGO Profile Models
======================================================
"""

from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import UserRole, OrganizationStatus


class User(Base):
    __tablename__ = "users"

    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.DONOR, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    memberships = relationship("OrganizationMembership", back_populates="user")
    donations = relationship("Donation", back_populates="donor")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    bio = Column(Text, nullable=True)
    phone_number = Column(String(32), nullable=True)
    avatar_url = Column(String(512), nullable=True)

    user = relationship("User", back_populates="profile")


class Organization(Base):
    __tablename__ = "organizations"

    name = Column(String(255), unique=True, nullable=False, index=True)
    registration_number = Column(String(100), unique=True, nullable=True)
    tax_id = Column(String(100), nullable=True)
    status = Column(SQLEnum(OrganizationStatus), default=OrganizationStatus.PENDING, nullable=False)
    website = Column(String(255), nullable=True)
    country = Column(String(3), nullable=False, default="USA")

    memberships = relationship("OrganizationMembership", back_populates="organization")
    ngo_profile = relationship("NGOProfile", back_populates="organization", uselist=False, cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organization")


class OrganizationMembership(Base):
    __tablename__ = "organization_memberships"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.NGO_MEMBER, nullable=False)

    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")


class NGOProfile(Base):
    __tablename__ = "ngo_profiles"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), unique=True, nullable=False)
    mission_statement = Column(Text, nullable=True)
    reputation_score = Column(Integer, default=100, nullable=False)
    completed_milestones = Column(Integer, default=0, nullable=False)
    rejected_milestones = Column(Integer, default=0, nullable=False)
    total_raised_minor_units = Column(Integer, default=0, nullable=False)

    organization = relationship("Organization", back_populates="ngo_profile")

