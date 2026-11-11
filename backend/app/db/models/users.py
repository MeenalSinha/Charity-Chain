"""
users.py - Identity, Authentication, Organization & Session Models
==================================================================
Implements models for User, UserSession, EmailVerificationToken,
PasswordResetToken, Organization, OrganizationMembership, and OrganizationInvitation.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Boolean, ForeignKey, Text, DateTime,
    UniqueConstraint, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, OrganizationStatus,
    AccountStatus, InvitationStatus
)


class User(Base):
    __tablename__ = "users"

    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    
    # Platform-wide role (DONOR, REVIEWER, PLATFORM_ADMIN)
    role = Column(SQLEnum(PlatformRole), default=PlatformRole.DONOR, nullable=False)
    
    # Account status & activation lifecycle
    account_status = Column(SQLEnum(AccountStatus), default=AccountStatus.PENDING_VERIFICATION, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified_at = Column(DateTime, nullable=True)
    
    # Security tracking
    password_changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    memberships = relationship("OrganizationMembership", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    verification_tokens = relationship("EmailVerificationToken", back_populates="user", cascade="all, delete-orphan")
    reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")
    donations = relationship("Donation", back_populates="donor")

    @property
    def platform_role(self) -> PlatformRole:
        """Alias for role adhering to PlatformRole terminology."""
        return self.role

    @platform_role.setter
    def platform_role(self, val: PlatformRole):
        self.role = val


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    bio = Column(Text, nullable=True)
    phone_number = Column(String(32), nullable=True)
    avatar_url = Column(String(512), nullable=True)

    user = relationship("User", back_populates="profile")


class UserSession(Base):
    """Persistent session tracking rotated refresh tokens and token families."""
    __tablename__ = "user_sessions"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    refresh_token_hash = Column(String(64), unique=True, nullable=False, index=True)
    token_family = Column(String(36), nullable=False, index=True)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    expires_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    revocation_reason = Column(String(255), nullable=True)

    user = relationship("User", back_populates="sessions")


class EmailVerificationToken(Base):
    """Cryptographic single-use token hash for email verification."""
    __tablename__ = "email_verification_tokens"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    is_used = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="verification_tokens")


class PasswordResetToken(Base):
    """Cryptographic single-use token hash for account recovery."""
    __tablename__ = "password_reset_tokens"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    is_used = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="reset_tokens")


class Organization(Base):
    __tablename__ = "organizations"

    name = Column(String(255), unique=True, nullable=False, index=True)
    registration_number = Column(String(100), unique=True, nullable=True)
    tax_id = Column(String(100), nullable=True)
    status = Column(SQLEnum(OrganizationStatus), default=OrganizationStatus.PENDING, nullable=False)
    website = Column(String(255), nullable=True)
    country = Column(String(3), nullable=False, default="USA")

    memberships = relationship("OrganizationMembership", back_populates="organization", cascade="all, delete-orphan")
    invitations = relationship("OrganizationInvitation", back_populates="organization", cascade="all, delete-orphan")
    ngo_profile = relationship("NGOProfile", back_populates="organization", uselist=False, cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organization")


class OrganizationMembership(Base):
    """Scoping a user to an organization with a discrete OrganizationRole."""
    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_org_membership_user_org"),
    )

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(OrganizationRole), default=OrganizationRole.MEMBER, nullable=False)

    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")


class OrganizationInvitation(Base):
    """Pending member invitation with secure token hash and expiration."""
    __tablename__ = "organization_invitations"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    inviter_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    invitee_email = Column(String(255), nullable=False, index=True)
    role = Column(SQLEnum(OrganizationRole), default=OrganizationRole.MEMBER, nullable=False)
    
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    is_revoked = Column(Boolean, default=False, nullable=False)
    status = Column(SQLEnum(InvitationStatus), default=InvitationStatus.PENDING, nullable=False)

    organization = relationship("Organization", back_populates="invitations")
    inviter = relationship("User")


class NGOProfile(Base):
    __tablename__ = "ngo_profiles"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), unique=True, nullable=False)
    mission_statement = Column(Text, nullable=True)
    reputation_score = Column(Integer, default=100, nullable=False)
    completed_milestones = Column(Integer, default=0, nullable=False)
    rejected_milestones = Column(Integer, default=0, nullable=False)
    total_raised_minor_units = Column(Integer, default=0, nullable=False)

    organization = relationship("Organization", back_populates="ngo_profile")
