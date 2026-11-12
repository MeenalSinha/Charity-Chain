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
    FINANCE_OPERATOR = "FINANCE_OPERATOR"
    FINANCE_APPROVER = "FINANCE_APPROVER"
    FINANCE_AUDITOR = "FINANCE_AUDITOR"


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
    """Lifecycle states for milestone-based outbound payouts."""
    REQUESTED = "REQUESTED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    SUBMITTED = "SUBMITTED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    HELD = "HELD"
    UNCLAIMED = "UNCLAIMED"
    RETURNED = "RETURNED"
    CANCELLED = "CANCELLED"
    REVERSED = "REVERSED"
    # Legacy alias
    SCHEDULED = "SCHEDULED"


PayoutStatus = DisbursementStatus


class PayoutRecipientStatus(str, Enum):
    """Verification and operational lifecycle of NGO payout destinations."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    UNDER_REVIEW = "UNDER_REVIEW"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SUSPENDED = "SUSPENDED"
    INACTIVE = "INACTIVE"


class RecipientType(str, Enum):
    """Supported payout rail recipient types."""
    PAYPAL_EMAIL = "PAYPAL_EMAIL"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    UPI = "UPI"


class DisbursementMethod(str, Enum):
    """Outbound disbursement payment rails."""
    PAYPAL_PAYOUTS = "PAYPAL_PAYOUTS"
    BANK_TRANSFER = "BANK_TRANSFER"
    CRYPTO = "CRYPTO"


class LedgerAccountType(str, Enum):
    """Core financial classification of double-entry ledger accounts."""
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"


class LedgerEntryType(str, Enum):
    """Categorical business transaction types producing journal entries."""
    DONATION_CAPTURE = "DONATION_CAPTURE"
    PROCESSOR_FEE = "PROCESSOR_FEE"
    DONATION_REFUND = "DONATION_REFUND"
    PAYOUT_COMMITMENT = "PAYOUT_COMMITMENT"
    PAYOUT_EXECUTION = "PAYOUT_EXECUTION"
    PAYOUT_FAILURE_REVERSAL = "PAYOUT_FAILURE_REVERSAL"
    PAYOUT_RETURN = "PAYOUT_RETURN"
    SETTLEMENT_RECONCILIATION = "SETTLEMENT_RECONCILIATION"
    MANUAL_ADJUSTMENT = "MANUAL_ADJUSTMENT"


class ReconciliationRunStatus(str, Enum):
    """Execution status of an automated reconciliation job."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_DISCREPANCIES = "COMPLETED_WITH_DISCREPANCIES"
    FAILED = "FAILED"


class ReconciliationStatus(str, Enum):
    """Reconciliation state of an individual transaction item."""
    MATCHED = "MATCHED"
    UNMATCHED = "UNMATCHED"
    DISCREPANCY = "DISCREPANCY"
    MANUALLY_RESOLVED = "MANUALLY_RESOLVED"
    IGNORED = "IGNORED"


class DiscrepancyType(str, Enum):
    """Classification of financial reconciliation mismatches."""
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    STATUS_MISMATCH = "STATUS_MISMATCH"
    MISSING_IN_LOCAL = "MISSING_IN_LOCAL"
    MISSING_IN_GATEWAY = "MISSING_IN_GATEWAY"
    FEE_MISMATCH = "FEE_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    TIMING_DIFFERENCE = "TIMING_DIFFERENCE"


class DiscrepancyResolution(str, Enum):
    """Resolution applied to an audited discrepancy."""
    AUTO_ADJUSTED = "AUTO_ADJUSTED"
    MANUALLY_RECONCILED = "MANUALLY_RECONCILED"
    PENDING_INVESTIGATION = "PENDING_INVESTIGATION"
    DISMISSED = "DISMISSED"


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_REVISION = "REQUEST_REVISION"


class ReviewCaseStatus(str, Enum):
    """Lifecycle status of a milestone human review case."""
    SUBMITTED = "SUBMITTED"
    ASSIGNED = "ASSIGNED"
    IN_REVIEW = "IN_REVIEW"
    AWAITING_SECOND_REVIEW = "AWAITING_SECOND_REVIEW"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    DISPUTED = "DISPUTED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ReviewAssignmentStatus(str, Enum):
    """Lifecycle status of an individual reviewer task assignment."""
    ASSIGNED = "ASSIGNED"
    ACCEPTED = "ACCEPTED"
    IN_PROGRESS = "IN_PROGRESS"
    DECIDED = "DECIDED"
    DECLINED = "DECLINED"
    REASSIGNED = "REASSIGNED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ReconciliationResult(str, Enum):
    """Reconciliation state between two independent review decisions."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


class DisputeStatus(str, Enum):
    """Formal lifecycle of an appeal or dispute against a milestone review decision."""
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    AWAITING_INFORMATION = "AWAITING_INFORMATION"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    WITHDRAWN = "WITHDRAWN"
    CLOSED = "CLOSED"


class DisputeCategory(str, Enum):
    """Categorical classification of formal reviewer disputes."""
    EVIDENCE_MISINTERPRETATION = "EVIDENCE_MISINTERPRETATION"
    PROCEDURAL_ERROR = "PROCEDURAL_ERROR"
    CONFLICT_OF_INTEREST = "CONFLICT_OF_INTEREST"
    TECHNICAL_VERIFICATION_ISSUE = "TECHNICAL_VERIFICATION_ISSUE"
    INCORRECT_ASSOCIATION = "INCORRECT_ASSOCIATION"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    REVIEWER_MISCONDUCT = "REVIEWER_MISCONDUCT"
    OTHER = "OTHER"


class DisputeResolution(str, Enum):
    """Authorized outcome of an adjudicated dispute."""
    UPHOLD_ORIGINAL_DECISION = "UPHOLD_ORIGINAL_DECISION"
    OVERTURN_ORIGINAL_DECISION = "OVERTURN_ORIGINAL_DECISION"
    REQUEST_NEW_EVIDENCE = "REQUEST_NEW_EVIDENCE"
    ORDER_RE_REVIEW = "ORDER_RE_REVIEW"
    DISMISS_DISPUTE = "DISMISS_DISPUTE"


class ReviewAuditEventType(str, Enum):
    """Immutable audit trail event types for human review governance."""
    CASE_CREATED = "CASE_CREATED"
    REVIEWER_ASSIGNED = "REVIEWER_ASSIGNED"
    ASSIGNMENT_ACCEPTED = "ASSIGNMENT_ACCEPTED"
    ASSIGNMENT_DECLINED = "ASSIGNMENT_DECLINED"
    CONFLICT_DECLARED = "CONFLICT_DECLARED"
    DECISION_SUBMITTED = "DECISION_SUBMITTED"
    DECISION_RECONCILED = "DECISION_RECONCILED"
    ESCALATED = "ESCALATED"
    ADJUDICATED = "ADJUDICATED"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    NGO_RESUBMITTED = "NGO_RESUBMITTED"
    DISPUTE_RAISED = "DISPUTE_RAISED"
    DISPUTE_MESSAGE_ADDED = "DISPUTE_MESSAGE_ADDED"
    DISPUTE_RESOLVED = "DISPUTE_RESOLVED"
    ADMIN_OVERRIDE = "ADMIN_OVERRIDE"


class CertificateType(str, Enum):
    """Centralized taxonomy of Proof-of-Impact certificate types."""
    CONTRIBUTION_IMPACT = "CONTRIBUTION_IMPACT"
    MILESTONE_IMPACT_ATTESTATION = "MILESTONE_IMPACT_ATTESTATION"
    DISBURSEMENT_ATTESTATION = "DISBURSEMENT_ATTESTATION"


class CertificateStatus(str, Enum):
    """Lifecycle states of cryptographically verifiable impact certificates."""
    PENDING_ELIGIBILITY = "PENDING_ELIGIBILITY"
    ISSUED = "ISSUED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"
    SUPERSEDED = "SUPERSEDED"


class CertificateRevocationReason(str, Enum):
    """Documented reason codes for certificate suspension or revocation."""
    FULL_REFUND = "FULL_REFUND"
    CHARGEBACK = "CHARGEBACK"
    REVIEW_OVERTURNED = "REVIEW_OVERTURNED"
    FRAUDULENT_EVIDENCE = "FRAUDULENT_EVIDENCE"
    INCORRECT_CLAIM = "INCORRECT_CLAIM"
    KEY_COMPROMISE = "KEY_COMPROMISE"
    ADMINISTRATIVE_CORRECTION = "ADMINISTRATIVE_CORRECTION"
    SUPERSEDED_BY_REVISION = "SUPERSEDED_BY_REVISION"


class ImpactLedgerEventType(str, Enum):
    """Cryptographically linked audit events on the Proof-of-Impact ledger."""
    ORGANIZATION_VERIFIED = "ORGANIZATION_VERIFIED"
    PROJECT_APPROVED = "PROJECT_APPROVED"
    PROJECT_ACTIVATED = "PROJECT_ACTIVATED"
    DONATION_CAPTURED = "DONATION_CAPTURED"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    EVIDENCE_VERSION_CREATED = "EVIDENCE_VERSION_CREATED"
    EVIDENCE_INTEGRITY_VERIFIED = "EVIDENCE_INTEGRITY_VERIFIED"
    AI_VERIFICATION_COMPLETED = "AI_VERIFICATION_COMPLETED"
    MILESTONE_REVIEW_SUBMITTED = "MILESTONE_REVIEW_SUBMITTED"
    FIRST_REVIEWER_DECISION = "FIRST_REVIEWER_DECISION"
    SECOND_REVIEWER_DECISION = "SECOND_REVIEWER_DECISION"
    DUAL_KEY_APPROVAL_FINALIZED = "DUAL_KEY_APPROVAL_FINALIZED"
    DISPUTE_RESOLVED = "DISPUTE_RESOLVED"
    MILESTONE_COMPLETED = "MILESTONE_COMPLETED"
    DISBURSEMENT_CONFIRMED = "DISBURSEMENT_CONFIRMED"
    FINANCIAL_RECONCILIATION_COMPLETED = "FINANCIAL_RECONCILIATION_COMPLETED"
    CERTIFICATE_ISSUED = "CERTIFICATE_ISSUED"
    CERTIFICATE_SUSPENDED = "CERTIFICATE_SUSPENDED"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    CERTIFICATE_SUPERSEDED = "CERTIFICATE_SUPERSEDED"


class SigningKeyStatus(str, Enum):
    """Operational status of Ed25519 issuer signing keys."""
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REVOKED = "REVOKED"


# -----------------------------------------------------------------------------
# Phase 10: Notifications, Webhooks & Communications
# -----------------------------------------------------------------------------

class NotificationCategory(str, Enum):
    """Categorization for in-app and transactional notifications."""
    ACCOUNT = "ACCOUNT"
    DONATIONS = "DONATIONS"
    PAYMENTS = "PAYMENTS"
    PROJECTS = "PROJECTS"
    MILESTONES = "MILESTONES"
    EVIDENCE = "EVIDENCE"
    VERIFICATION = "VERIFICATION"
    DISBURSEMENTS = "DISBURSEMENTS"
    CERTIFICATES = "CERTIFICATES"
    SECURITY = "SECURITY"
    SYSTEM = "SYSTEM"


class NotificationChannel(str, Enum):
    """Delivery mechanism channels for notifications."""
    IN_APP = "IN_APP"
    EMAIL = "EMAIL"


class OutboxStatus(str, Enum):
    """Processing state for transactional event outbox records."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


class WebhookDeliveryStatus(str, Enum):
    """Delivery attempt state for external webhook dispatches."""
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


class WebhookSubscriptionStatus(str, Enum):
    """Lifecycle and operational state for external webhook subscriptions."""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    INACTIVE = "INACTIVE"


class DomainEventType(str, Enum):
    """Standardized event catalog across CharityChain domain contexts."""
    # User and Organization
    USER_REGISTERED = "user.registered"
    EMAIL_VERIFICATION_COMPLETED = "user.email_verified"
    ORGANIZATION_REGISTERED = "organization.registered"
    ORGANIZATION_VERIFICATION_APPROVED = "organization.verified"
    ORGANIZATION_VERIFICATION_REJECTED = "organization.rejected"
    ORGANIZATION_SUSPENDED = "organization.suspended"

    # Project and Milestone
    PROJECT_CREATED = "project.created"
    PROJECT_PUBLISHED = "project.published"
    PROJECT_APPROVED = "project.approved"
    PROJECT_SUSPENDED = "project.suspended"
    MILESTONE_CREATED = "milestone.created"
    MILESTONE_UPDATED = "milestone.updated"
    MILESTONE_COMPLETED = "milestone.completed"

    # Donation and Payment
    DONATION_CREATED = "donation.created"
    PAYMENT_AUTHORIZED = "payment.authorized"
    PAYMENT_CAPTURED = "payment.captured"
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_REFUNDED = "payment.refunded"
    PAYMENT_DISPUTED = "payment.disputed"

    # Evidence and AI Verification
    EVIDENCE_SUBMITTED = "evidence.submitted"
    EVIDENCE_PROCESSING_STARTED = "evidence.processing_started"
    AI_VERIFICATION_COMPLETED = "ai_verification.completed"
    AI_VERIFICATION_FAILED = "ai_verification.failed"
    ADDITIONAL_EVIDENCE_REQUESTED = "evidence.additional_requested"
    EVIDENCE_APPROVED = "evidence.approved"
    EVIDENCE_REJECTED = "evidence.rejected"

    # Review and Governance
    REVIEW_ASSIGNED = "review.assigned"
    REVIEW_APPROVED = "review.approved"
    REVIEW_REJECTED = "review.rejected"
    DUAL_KEY_APPROVAL_COMPLETED = "review.dual_key_completed"
    DISPUTE_OPENED = "dispute.opened"
    DISPUTE_RESOLVED = "dispute.resolved"

    # Disbursement
    DISBURSEMENT_REQUESTED = "disbursement.requested"
    DISBURSEMENT_APPROVED = "disbursement.approved"
    DISBURSEMENT_PROCESSING = "disbursement.processing"
    DISBURSEMENT_COMPLETED = "disbursement.completed"
    DISBURSEMENT_FAILED = "disbursement.failed"

    # Impact and Certificate
    CERTIFICATE_ISSUED = "certificate.issued"
    CERTIFICATE_SUSPENDED = "certificate.suspended"
    CERTIFICATE_REINSTATED = "certificate.reinstated"
    CERTIFICATE_REVOKED = "certificate.revoked"
    CERTIFICATE_SUPERSEDED = "certificate.superseded"
    IMPACT_LEDGER_CHECKPOINT_CREATED = "impact_ledger.checkpoint_created"
    IMPACT_LEDGER_INTEGRITY_FAILURE = "impact_ledger.integrity_failure"



