"""
schemas.py - Common Standardized API Request & Response Schemas
==============================================================
Defines consistent envelopes, pagination, and error payloads across all endpoints.
"""

from typing import Generic, TypeVar, Optional, List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class StandardResponse(BaseModel, Generic[DataT]):
    """Standard success response wrapper."""
    success: bool = True
    data: DataT
    message: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StandardErrorResponse(BaseModel):
    """Standard error response wrapper."""
    success: bool = False
    error: ErrorDetail
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class PaginatedResponse(BaseModel, Generic[DataT]):
    """Paginated list response wrapper."""
    success: bool = True
    data: List[DataT]
    pagination: PaginationMeta
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)



# ============================================================================
# PHASE 2 IDENTITY & AUTHENTICATION SCHEMAS
# ============================================================================

from backend.app.core.constants import PlatformRole, OrganizationRole, AccountStatus, InvitationStatus


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    password_confirm: str = Field(..., min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: PlatformRole
    account_status: AccountStatus
    requires_email_verification: bool = True


class VerifyEmailRequest(BaseModel):
    token: str = Field(..., min_length=10)


class ResendVerificationRequest(BaseModel):
    email: str = Field(..., max_length=255)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str
    full_name: str
    role: PlatformRole


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., max_length=255)


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)
    new_password_confirm: str = Field(..., min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    new_password_confirm: str = Field(..., min_length=8, max_length=128)


class UserProfileResponse(BaseModel):
    bio: Optional[str] = None
    phone_number: Optional[str] = None
    avatar_url: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: PlatformRole
    account_status: AccountStatus
    is_verified: bool
    email_verified_at: Optional[datetime] = None
    created_at: datetime
    profile: Optional[UserProfileResponse] = None


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)


class UpdateProfileRequest(BaseModel):
    bio: Optional[str] = Field(None, max_length=2000)
    phone_number: Optional[str] = Field(None, max_length=32)
    avatar_url: Optional[str] = Field(None, max_length=512)


class SessionResponse(BaseModel):
    id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    last_used_at: datetime
    expires_at: datetime
    is_revoked: bool


class OrganizationMembershipResponse(BaseModel):
    id: str
    organization_id: str
    organization_name: str
    role: OrganizationRole
    created_at: datetime


class MemberDetailResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: OrganizationRole
    joined_at: datetime


class CreateInvitationRequest(BaseModel):
    invitee_email: str = Field(..., max_length=255)
    role: OrganizationRole = OrganizationRole.MEMBER


class OrganizationInvitationResponse(BaseModel):
    id: str
    organization_id: str
    invitee_email: str
    role: OrganizationRole
    status: InvitationStatus
    expires_at: datetime
    created_at: datetime


class UpdateMemberRoleRequest(BaseModel):
    role: OrganizationRole


class UpdateUserStatusRequest(BaseModel):
    status: AccountStatus


class UpdateUserRoleRequest(BaseModel):
    role: PlatformRole
