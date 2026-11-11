"""
constants.py - Centralized Domain Enums & System Constants
==========================================================
Eliminates magic strings and scattered literal definitions.
"""

from enum import Enum


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class PlatformRole(str, Enum):
    """Platform-wide system roles."""
    DONOR = "DONOR"
    REVIEWER = "REVIEWER"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"


class OrganizationRole(str, Enum):
    """Organization-scoped membership roles."""
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"


class AccountStatus(str, Enum):
    """User account lifecycle status."""
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    ACTIVE = "ACTIVE"
    LOCKED = "LOCKED"
    SUSPENDED = "SUSPENDED"
    DEACTIVATED = "DEACTIVATED"


class InvitationStatus(str, Enum):
    """Organization invitation status."""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class TokenType(str, Enum):
    """Cryptographic security token types."""
    EMAIL_VERIFICATION = "EMAIL_VERIFICATION"
    PASSWORD_RESET = "PASSWORD_RESET"
    REFRESH = "REFRESH"


class UserRole(str, Enum):
    """Legacy compatibility enum mapping both platform and legacy roles."""
    DONOR = "DONOR"
    NGO_ADMIN = "NGO_ADMIN"
    NGO_MEMBER = "NGO_MEMBER"
    REVIEWER = "REVIEWER"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"


class OrganizationType(str, Enum):
    """Legal classifications for nonprofit organizations."""
    NON_PROFIT = "NON_PROFIT"
    CHARITY = "CHARITY"
    FOUNDATION = "FOUNDATION"
    NGO = "NGO"
    TRUST = "TRUST"
    SOCIAL_ENTERPRISE = "SOCIAL_ENTERPRISE"
    OTHER = "OTHER"


class OrganizationStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING = "PENDING"
    UNDER_REVIEW = "UNDER_REVIEW"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"


class OrganizationDocumentType(str, Enum):
    """Supported onboarding verification document categories."""
    REGISTRATION_CERTIFICATE = "REGISTRATION_CERTIFICATE"
    TAX_EXEMPTION_CERTIFICATE = "TAX_EXEMPTION_CERTIFICATE"
    NONPROFIT_STATUS_DOCUMENTATION = "NONPROFIT_STATUS_DOCUMENTATION"
    GOVERNING_DOCUMENTS = "GOVERNING_DOCUMENTS"
    ADDRESS_VERIFICATION = "ADDRESS_VERIFICATION"
    AUTHORIZED_REPRESENTATIVE_DOCUMENTATION = "AUTHORIZED_REPRESENTATIVE_DOCUMENTATION"
    OTHER_SUPPORTING_DOCUMENTS = "OTHER_SUPPORTING_DOCUMENTS"


class DocumentReviewStatus(str, Enum):
    """Review status for verification documents."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISION_REQUIRED = "REVISION_REQUIRED"


class DocumentAccessClassification(str, Enum):
    """Access classification for stored documents."""
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class ProjectVisibility(str, Enum):
    """Visibility scope for charitable projects."""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"


class ProjectStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class MilestoneStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    CANCELLED = "CANCELLED"


class EvidenceStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class VerificationCheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    FLAGGED = "FLAGGED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PENDING = "PENDING"
    ERROR = "ERROR"


class OverallAssessmentStatus(str, Enum):
    """Categorical outcome of multi-indicator AI verification."""
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    NEEDS_FURTHER_REVIEW = "NEEDS_FURTHER_REVIEW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    PROCESSING_FAILED = "PROCESSING_FAILED"


class VerificationJobStatus(str, Enum):
    """Lifecycle status of asynchronous verification jobs."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


class VerificationCheckType(str, Enum):
    """Supported computer vision and forensic check types."""
    YOLO_OBJECTS = "YOLO_OBJECTS"
    ELA_TAMPER = "ELA_TAMPER"
    PHASH_SIMILARITY = "PHASH_SIMILARITY"
    SSIM_SIMILARITY = "SSIM_SIMILARITY"
    EXIF_METADATA = "EXIF_METADATA"
    GPS_CONSISTENCY = "GPS_CONSISTENCY"
    PROGRESS_COMPARISON = "PROGRESS_COMPARISON"
    LLM_EXPLAINABILITY = "LLM_EXPLAINABILITY"
    FILE_INTEGRITY = "FILE_INTEGRITY"


class DonationStatus(str, Enum):
    """Centralized lifecycle states for a charitable donation intent."""
    CREATED = "CREATED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAID = "PAID"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"
    REFUNDED = "REFUNDED"


class WebhookProcessingStatus(str, Enum):
    """Lifecycle status for persisted incoming payment webhooks."""
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    DUPLICATE = "DUPLICATE"
    FAILED = "FAILED"
    IGNORED = "IGNORED"


class WebhookSignatureStatus(str, Enum):
    """Cryptographic signature validation status of payment webhooks."""
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PaymentOrderStatus(str, Enum):
    CREATED = "CREATED"
    SAVED = "SAVED"
    APPROVED = "APPROVED"
    VOIDED = "VOIDED"
    COMPLETED = "COMPLETED"


class PaymentAuthorizationStatus(str, Enum):
    CREATED = "CREATED"
    CAPTURED = "CAPTURED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    PARTIALLY_CAPTURED = "PARTIALLY_CAPTURED"
    VOIDED = "VOIDED"


class PaymentCaptureStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"
    REFUNDED = "REFUNDED"


class PaymentRefundStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DisbursementStatus(str, Enum):
    SCHEDULED = "SCHEDULED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_REVISION = "REQUEST_REVISION"

