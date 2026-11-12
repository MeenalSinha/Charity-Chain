"""
__init__.py - Expose all models for SQLAlchemy Base.metadata discovery
"""

from backend.app.db.base import Base
from backend.app.db.models.users import (
    User, UserProfile, Organization, OrganizationMembership, NGOProfile,
    UserSession, EmailVerificationToken, PasswordResetToken, OrganizationInvitation,
    OrganizationDocument
)
from backend.app.db.models.projects import ProjectCategory, Project, Milestone, ProjectLocation
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentAuthorization, PaymentCapture,
    PaymentRefund, WebhookEvent, Disbursement,
    PayoutRecipient, PayoutBatch, PayoutApproval,
    ReconciliationRun, ReconciliationItem, SettlementRecord, ProcessorFeeRecord
)
from backend.app.db.models.ledger import (
    LedgerAccount, JournalEntry, JournalLine
)
from backend.app.db.models.evidence import (
    EvidenceFile, VerificationReport, VerificationCheckRecord,
    VerificationJob, EvidenceAuditEvent, compute_audit_hash
)
from backend.app.db.models.reviews_impact import (
    ReviewCase, ReviewCaseEvidence, ReviewTask, ReviewDecisionRecord,
    ReviewDispute, ReviewDisputeMessage, ReviewAuditEvent,
    compute_review_audit_hash, ImpactCertificate, AuditEvent
)

__all__ = [
    "Base",
    "User",
    "UserProfile",
    "Organization",
    "OrganizationMembership",
    "NGOProfile",
    "UserSession",
    "EmailVerificationToken",
    "PasswordResetToken",
    "OrganizationInvitation",
    "OrganizationDocument",
    "ProjectCategory",
    "Project",
    "Milestone",
    "ProjectLocation",
    "Donation",
    "PaymentOrder",
    "PaymentAuthorization",
    "PaymentCapture",
    "PaymentRefund",
    "WebhookEvent",
    "Disbursement",
    "PayoutRecipient",
    "PayoutBatch",
    "PayoutApproval",
    "ReconciliationRun",
    "ReconciliationItem",
    "SettlementRecord",
    "ProcessorFeeRecord",
    "LedgerAccount",
    "JournalEntry",
    "JournalLine",
    "EvidenceFile",
    "VerificationReport",
    "VerificationCheckRecord",
    "VerificationJob",
    "EvidenceAuditEvent",
    "compute_audit_hash",
    "ReviewCase",
    "ReviewCaseEvidence",
    "ReviewTask",
    "ReviewDecisionRecord",
    "ReviewDispute",
    "ReviewDisputeMessage",
    "ReviewAuditEvent",
    "compute_review_audit_hash",
    "ImpactCertificate",
    "AuditEvent"
]

