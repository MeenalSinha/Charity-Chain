"""
schemas.py - Common Standardized API Request & Response Schemas
==============================================================
Defines consistent envelopes, pagination, and error payloads across all endpoints.
"""

from typing import Generic, TypeVar, Optional, List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

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

    @property
    def items(self) -> List[DataT]:
        return self.data

    @property
    def total_count(self) -> int:
        return self.pagination.total_items


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


# ============================================================================
# PHASE 3 ORGANIZATION, PROJECT & MILESTONE SCHEMAS
# ============================================================================

from backend.app.core.constants import (
    OrganizationType, OrganizationStatus, OrganizationDocumentType,
    DocumentReviewStatus, DocumentAccessClassification,
    ProjectStatus, MilestoneStatus, ProjectVisibility
)


# --- Organization Onboarding & Profile Schemas ---

class CreateOrganizationRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    legal_name: Optional[str] = Field(None, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    organization_type: OrganizationType = OrganizationType.NON_PROFIT
    country: str = Field(default="USA", max_length=3)
    registration_number: Optional[str] = Field(None, max_length=100)
    registration_authority: Optional[str] = Field(None, max_length=255)
    registration_date: Optional[datetime] = None
    tax_id: Optional[str] = Field(None, max_length=100)
    tax_identification_number: Optional[str] = Field(None, max_length=100)
    country_of_registration: Optional[str] = Field(None, max_length=3)
    classification: Optional[str] = Field(None, max_length=100)
    registered_address: Optional[str] = None
    operating_address: Optional[str] = None
    primary_operating_region: Optional[str] = None
    official_email: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    official_phone: Optional[str] = Field(None, max_length=32)
    contact_phone: Optional[str] = Field(None, max_length=32)
    website: Optional[str] = Field(None, max_length=255)
    website_url: Optional[str] = Field(None, max_length=255)
    mission_statement: Optional[str] = None
    vision: Optional[str] = None
    vision_statement: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    operating_regions: Optional[List[str]] = None
    beneficiary_categories: Optional[List[str]] = None
    years_of_operation: Optional[int] = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("name") and data.get("legal_name"):
                data["name"] = data["legal_name"]
            elif not data.get("legal_name") and data.get("name"):
                data["legal_name"] = data["name"]
            if not data.get("country") and data.get("country_of_registration"):
                data["country"] = data["country_of_registration"]
            if not data.get("operating_regions") and data.get("primary_operating_region"):
                data["operating_regions"] = [data["primary_operating_region"]]
            if not data.get("tax_id") and data.get("tax_identification_number"):
                data["tax_id"] = data["tax_identification_number"]
            if not data.get("official_email") and data.get("contact_email"):
                data["official_email"] = data["contact_email"]
            if not data.get("official_phone") and data.get("contact_phone"):
                data["official_phone"] = data["contact_phone"]
            if not data.get("website") and data.get("website_url"):
                data["website"] = data["website_url"]
            if not data.get("vision") and data.get("vision_statement"):
                data["vision"] = data["vision_statement"]
        return data


class UpdateOrganizationRequest(BaseModel):
    legal_name: Optional[str] = Field(None, max_length=255)
    display_name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    organization_type: Optional[OrganizationType] = None
    country: Optional[str] = Field(None, max_length=3)
    registration_number: Optional[str] = Field(None, max_length=100)
    registration_authority: Optional[str] = Field(None, max_length=255)
    registration_date: Optional[datetime] = None
    tax_id: Optional[str] = Field(None, max_length=100)
    classification: Optional[str] = Field(None, max_length=100)
    registered_address: Optional[str] = None
    operating_address: Optional[str] = None
    official_email: Optional[str] = Field(None, max_length=255)
    official_phone: Optional[str] = Field(None, max_length=32)
    website: Optional[str] = Field(None, max_length=255)
    mission_statement: Optional[str] = None
    vision: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    operating_regions: Optional[List[str]] = None
    beneficiary_categories: Optional[List[str]] = None
    years_of_operation: Optional[int] = Field(None, ge=0)


class OrganizationPublicResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    organization_type: OrganizationType
    country: str
    website: Optional[str] = None
    mission_statement: Optional[str] = None
    vision: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    operating_regions: Optional[List[str]] = None
    years_of_operation: int = 0
    status: OrganizationStatus
    verified_at: Optional[datetime] = None
    created_at: datetime


class OrganizationDetailResponse(BaseModel):
    id: str
    name: str
    legal_name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    organization_type: OrganizationType
    registration_number: Optional[str] = None
    registration_authority: Optional[str] = None
    registration_date: Optional[datetime] = None
    tax_id: Optional[str] = None
    classification: Optional[str] = None
    registered_address: Optional[str] = None
    operating_address: Optional[str] = None
    country: str
    official_email: Optional[str] = None
    official_phone: Optional[str] = None
    website: Optional[str] = None
    mission_statement: Optional[str] = None
    vision: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    operating_regions: Optional[List[str]] = None
    beneficiary_categories: Optional[List[str]] = None
    years_of_operation: int = 0
    status: OrganizationStatus
    submitted_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    suspension_reason: Optional[str] = None
    revision_notes: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewed_by_id: Optional[str] = None
    created_at: datetime


class OrganizationDecisionRequest(BaseModel):
    decision: str = Field(..., description="APPROVE, REJECT, or REQUEST_REVISION")
    reason: str = Field(..., min_length=3, max_length=1000)
    notes: Optional[str] = None
    review_notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("reason") and data.get("review_notes"):
                data["reason"] = data["review_notes"]
            elif not data.get("review_notes") and data.get("reason"):
                data["review_notes"] = data["reason"]
            if not data.get("notes") and data.get("reason"):
                data["notes"] = data["reason"]
        return data


class OrganizationSuspensionRequest(BaseModel):
    reason: str = Field(..., min_length=5, max_length=1000)


# --- Document Schemas ---

class OrganizationDocumentResponse(BaseModel):
    id: str
    organization_id: str
    document_type: OrganizationDocumentType
    original_filename: str
    mime_type: str
    file_size_bytes: int
    file_hash_sha256: str
    version_number: int
    review_status: DocumentReviewStatus
    reviewer_comments: Optional[str] = None
    access_classification: DocumentAccessClassification
    created_at: datetime


class DocumentDecisionRequest(BaseModel):
    decision: DocumentReviewStatus
    comments: Optional[str] = None


# --- Project Category Schemas ---

class CreateProjectCategoryRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = None
    slug: Optional[str] = Field(None, max_length=100)


class UpdateProjectCategoryRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ProjectCategoryResponse(BaseModel):
    id: str
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    is_active: bool
    created_at: datetime


# --- Project & Milestone Schemas ---

class CreateProjectRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    summary: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None)
    organization_id: Optional[str] = None
    org_id: Optional[str] = None
    category_id: Optional[str] = None
    objectives: Optional[str] = None
    project_type: str = "CHARITABLE_PROJECT"
    public_visibility: ProjectVisibility = ProjectVisibility.PUBLIC
    target_amount_minor_units: Optional[int] = Field(None, description="Funding target in minor units (cents)")
    funding_target_minor: Optional[int] = Field(None, description="Funding target in minor units")
    currency: str = Field(default="USD", max_length=3)
    target_beneficiaries: Optional[str] = None
    expected_beneficiaries: Optional[int] = None
    beneficiary_location: Optional[str] = None
    primary_operating_region: Optional[str] = None
    expected_outcomes: Optional[str] = None
    deliverables: Optional[str] = None
    impact_metrics: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    country: Optional[str] = "USA"
    state_region: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_meters: Optional[float] = Field(default=1000.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("organization_id") and data.get("org_id"):
                data["organization_id"] = data["org_id"]
            elif not data.get("org_id") and data.get("organization_id"):
                data["org_id"] = data["organization_id"]
            if not data.get("description") and data.get("summary"):
                data["description"] = data["summary"]
            elif not data.get("summary") and data.get("description"):
                data["summary"] = data["description"][:500]
            if not data.get("target_amount_minor_units") and data.get("funding_target_minor"):
                data["target_amount_minor_units"] = data["funding_target_minor"]
            elif not data.get("funding_target_minor") and data.get("target_amount_minor_units"):
                data["funding_target_minor"] = data["target_amount_minor_units"]
            if not data.get("state_region") and data.get("primary_operating_region"):
                data["state_region"] = data["primary_operating_region"]
            if not data.get("city") and data.get("beneficiary_location"):
                data["city"] = data["beneficiary_location"]
            if not data.get("target_beneficiaries") and data.get("expected_beneficiaries"):
                data["target_beneficiaries"] = str(data["expected_beneficiaries"])
        return data


class UpdateProjectRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    summary: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, min_length=10)
    category_id: Optional[str] = None
    objectives: Optional[str] = None
    public_visibility: Optional[ProjectVisibility] = None
    target_amount_minor_units: Optional[int] = Field(None, gt=0)
    currency: Optional[str] = Field(None, max_length=3)
    target_beneficiaries: Optional[str] = None
    expected_outcomes: Optional[str] = None
    deliverables: Optional[str] = None
    impact_metrics: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    country: Optional[str] = None
    state_region: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_meters: Optional[float] = Field(None, gt=0)


class ProjectValidationIssue(BaseModel):
    field: str
    code: str
    message: str


class ProjectValidationResponse(BaseModel):
    valid: bool
    can_submit: bool
    issues: List[ProjectValidationIssue]

    @property
    def is_valid(self) -> bool:
        return self.valid


class ProjectDecisionRequest(BaseModel):
    decision: str = Field(..., description="APPROVE, REJECT, or REQUEST_REVISION")
    reason: str = Field(..., min_length=3, max_length=1000)
    notes: Optional[str] = None
    review_notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("reason") and data.get("review_notes"):
                data["reason"] = data["review_notes"]
            elif not data.get("review_notes") and data.get("reason"):
                data["review_notes"] = data["reason"]
            if not data.get("notes") and data.get("reason"):
                data["notes"] = data["reason"]
        return data


class CreateMilestoneRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    description: Optional[str] = Field(None)
    sequence_number: Optional[int] = Field(None, ge=1)
    target_amount_minor_units: Optional[int] = Field(None, ge=0)
    funding_target_minor: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=3)
    objective: Optional[str] = None
    deliverables: Optional[str] = None
    expected_outcome: Optional[str] = None
    success_criteria: Optional[str] = None
    required_evidence_desc: Optional[str] = None
    planned_start_date: Optional[datetime] = None
    planned_completion_date: Optional[datetime] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    target_latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    target_longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_meters: Optional[float] = Field(default=1000.0, gt=0)
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("description") and data.get("objective"):
                data["description"] = data["objective"]
            elif not data.get("objective") and data.get("description"):
                data["objective"] = data["description"]
            if data.get("target_amount_minor_units") is None and data.get("funding_target_minor") is not None:
                data["target_amount_minor_units"] = data["funding_target_minor"]
            elif data.get("funding_target_minor") is None and data.get("target_amount_minor_units") is not None:
                data["funding_target_minor"] = data["target_amount_minor_units"]
            if data.get("latitude") is None and data.get("target_latitude") is not None:
                data["latitude"] = data["target_latitude"]
            elif data.get("target_latitude") is None and data.get("latitude") is not None:
                data["target_latitude"] = data["latitude"]
            if data.get("longitude") is None and data.get("target_longitude") is not None:
                data["longitude"] = data["target_longitude"]
            elif data.get("target_longitude") is None and data.get("longitude") is not None:
                data["target_longitude"] = data["longitude"]
        return data


class UpdateMilestoneRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    description: Optional[str] = Field(None, min_length=5)
    target_amount_minor_units: Optional[int] = Field(None, ge=0)
    objective: Optional[str] = None
    deliverables: Optional[str] = None
    expected_outcome: Optional[str] = None
    success_criteria: Optional[str] = None
    required_evidence_desc: Optional[str] = None
    planned_start_date: Optional[datetime] = None
    planned_completion_date: Optional[datetime] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    radius_meters: Optional[float] = Field(None, gt=0)
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class MilestoneResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str
    sequence_number: int
    target_amount_minor_units: int
    raised_amount_minor_units: int
    currency: str
    status: MilestoneStatus
    objective: Optional[str] = None
    deliverables: Optional[str] = None
    expected_outcome: Optional[str] = None
    success_criteria: Optional[str] = None
    required_evidence_desc: Optional[str] = None
    planned_start_date: Optional[datetime] = None
    planned_completion_date: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_meters: float = 1000.0
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    created_at: datetime

    @property
    def funding_target_minor(self) -> int:
        return self.target_amount_minor_units

    @property
    def target_latitude(self) -> Optional[float]:
        return self.latitude

    @property
    def target_longitude(self) -> Optional[float]:
        return self.longitude


class ProjectResponse(BaseModel):
    id: str
    organization_id: str
    organization_name: Optional[str] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    title: str
    summary: Optional[str] = None
    description: str
    objectives: Optional[str] = None
    project_type: str
    public_visibility: ProjectVisibility
    status: ProjectStatus
    target_amount_minor_units: int
    raised_amount_minor_units: int
    currency: str
    target_beneficiaries: Optional[str] = None
    expected_outcomes: Optional[str] = None
    deliverables: Optional[str] = None
    impact_metrics: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    country: Optional[str] = None
    state_region: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_meters: float = 1000.0
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    revision_notes: Optional[str] = None
    milestone_count: int = 0
    milestones: Optional[List[MilestoneResponse]] = None
    created_at: datetime

    @property
    def funding_target_minor(self) -> int:
        return self.target_amount_minor_units

    @property
    def funding_raised_minor(self) -> int:
        return self.raised_amount_minor_units


# ============================================================================
# PHASE 4 DONATION, PAYMENT & REFUND SCHEMAS
# ============================================================================

from backend.app.core.constants import DonationStatus, PaymentOrderStatus, PaymentCaptureStatus, PaymentRefundStatus


class CreateDonationRequest(BaseModel):
    project_id: str
    milestone_id: str
    amount_minor_units: int = Field(..., gt=0, description="Contribution amount in integer minor units (e.g. 2500 for $25.00)")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    donor_name: Optional[str] = Field(None, max_length=255)
    donor_email: Optional[str] = Field(None, max_length=255)
    is_anonymous: bool = Field(default=False)
    return_url: Optional[str] = None
    cancel_url: Optional[str] = None


class DonationResponse(BaseModel):
    id: str
    donor_id: Optional[str] = None
    project_id: str
    milestone_id: str
    amount_minor_units: int
    currency: str
    status: DonationStatus
    donor_name: Optional[str] = None
    is_anonymous: bool = False
    order_id: Optional[str] = None
    provider_order_id: Optional[str] = None
    approval_url: Optional[str] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class CapturePaymentRequest(BaseModel):
    idempotency_key: Optional[str] = None


class CapturePaymentResponse(BaseModel):
    capture_id: str
    order_id: str
    provider_capture_id: str
    status: str
    amount_minor_units: int
    currency: str
    donation_status: str


class CreateRefundRequest(BaseModel):
    amount_minor_units: int = Field(..., gt=0, description="Refund amount in minor units")
    reason: str = Field(default="Donor requested refund", min_length=3, max_length=255)


class RefundResponse(BaseModel):
    refund_id: str
    donation_id: str
    provider_refund_id: str
    amount_minor_units: int
    currency: str
    status: str
    donation_status: str
    remaining_refundable_minor_units: int


class MilestoneFundingSummary(BaseModel):
    milestone_id: str
    sequence_number: int
    title: str
    target_amount_minor_units: int
    gross_received_minor_units: int
    refunded_minor_units: int
    net_received_minor_units: int
    percentage_funded: float


class ProjectFundingSummaryResponse(BaseModel):
    project_id: str
    project_title: str
    currency: str
    target_amount_minor_units: int
    gross_received_minor_units: int
    refunded_minor_units: int
    net_received_minor_units: int
    percentage_funded: float
    milestones: List[MilestoneFundingSummary]


# =====================================================================
# Phase 5: Evidence Management Schemas
# =====================================================================

class EvidenceAuditEventResponse(BaseModel):
    id: str
    evidence_id: str
    actor_id: Optional[str] = None
    event_type: str
    details_json: Optional[Dict[str, Any]] = None
    previous_event_hash: str
    current_event_hash: str
    created_at: datetime


class EvidenceFileResponse(BaseModel):
    id: str
    milestone_id: str
    uploader_id: Optional[str] = None
    original_filename: Optional[str] = None
    storage_key: Optional[str] = None
    storage_provider: str
    file_path: str
    file_hash: str
    content_address: str
    mime_type: str
    file_size_bytes: int
    version_number: int
    parent_evidence_id: Optional[str] = None
    status: str
    is_archived: bool
    archived_at: Optional[datetime] = None
    quarantine_reason: Optional[str] = None
    created_at: datetime
    download_url: Optional[str] = None


class EvidenceBatchUploadResponse(BaseModel):
    milestone_id: str
    total_uploaded: int
    items: List[EvidenceFileResponse]


class EvidenceIntegrityResponse(BaseModel):
    evidence_id: str
    expected_hash: str
    computed_hash: str
    is_valid: bool
    checked_at: datetime
    storage_provider: str


class EvidenceVersionHistoryResponse(BaseModel):
    evidence_id: str
    current_version: int
    versions: List[EvidenceFileResponse]


class EvidenceAuditLogResponse(BaseModel):
    evidence_id: str
    is_chain_intact: bool
    total_events: int
    events: List[EvidenceAuditEventResponse]


class ArchiveEvidenceRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=255)



