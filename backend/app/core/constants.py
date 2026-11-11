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


class UserRole(str, Enum):
    DONOR = "DONOR"
    NGO_ADMIN = "NGO_ADMIN"
    NGO_MEMBER = "NGO_MEMBER"
    REVIEWER = "REVIEWER"
    PLATFORM_ADMIN = "PLATFORM_ADMIN"


class OrganizationStatus(str, Enum):
    PENDING = "PENDING"
    UNDER_REVIEW = "UNDER_REVIEW"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"


class ProjectStatus(str, Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
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
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PENDING = "PENDING"


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

