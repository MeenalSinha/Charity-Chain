"""
financial.py - Currency-Safe Financial Domain Models
====================================================
Maintains strict separation between:
1. Donated Intent (Donation)
2. Payment Orders (PaymentOrder)
3. Pre-Authorization Holds (PaymentAuthorization)
4. Payment Captures (PaymentCapture)
5. Reversals/Refunds (PaymentRefund)
6. Webhook Audit Events (WebhookEvent)
7. Actual NGO Payouts (Disbursement)
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import (
    PaymentOrderStatus, PaymentAuthorizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus,
    DonationStatus, WebhookProcessingStatus, WebhookSignatureStatus,
    PayoutRecipientStatus, RecipientType, DisbursementMethod,
    ReconciliationRunStatus, ReconciliationStatus, DiscrepancyType,
    DiscrepancyResolution
)


class Donation(Base):
    __tablename__ = "donations"

    donor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(DonationStatus), default=DonationStatus.CREATED, nullable=False, index=True)
    impact_tokens_earned = Column(Integer, default=0, nullable=False)

    # Phase 4 Donor Attribution & Lifecycle
    donor_name = Column(String(255), nullable=True)
    donor_email = Column(String(255), nullable=True)
    is_anonymous = Column(Boolean, default=False, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    donor = relationship("User", back_populates="donations")
    project = relationship("Project", back_populates="donations")
    milestone = relationship("Milestone", back_populates="donations")
    orders = relationship("PaymentOrder", back_populates="donation", cascade="all, delete-orphan")
    certificates = relationship("ImpactCertificate", back_populates="donation")

    @property
    def certificate(self):
        """Returns the latest active or issued certificate (for backward compatibility)."""
        if self.certificates:
            active = [c for c in self.certificates if getattr(c, "status", "ACTIVE") in ("ACTIVE", "ISSUED")]
            return active[-1] if active else self.certificates[-1]
        return None


class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    donation_id = Column(String(36), ForeignKey("donations.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String(50), nullable=False, default="PAYPAL")
    provider_order_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentOrderStatus), default=PaymentOrderStatus.CREATED, nullable=False, index=True)

    # Phase 4 Idempotency & Provider Lifecycle
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)
    approval_url = Column(String(512), nullable=True)
    failure_code = Column(String(100), nullable=True)
    failure_reason = Column(Text, nullable=True)

    donation = relationship("Donation", back_populates="orders")
    authorizations = relationship("PaymentAuthorization", back_populates="order")
    captures = relationship("PaymentCapture", back_populates="order")


class PaymentAuthorization(Base):
    """Represents a temporary card hold. NOT an escrow account."""
    __tablename__ = "payment_authorizations"

    order_id = Column(String(36), ForeignKey("payment_orders.id", ondelete="CASCADE"), nullable=False, index=True)
    provider_authorization_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentAuthorizationStatus), default=PaymentAuthorizationStatus.CREATED, nullable=False)
    expires_at = Column(DateTime, nullable=True)

    order = relationship("PaymentOrder", back_populates="authorizations")
    captures = relationship("PaymentCapture", back_populates="authorization")


class PaymentCapture(Base):
    """Represents actual settlement of funds from donor."""
    __tablename__ = "payment_captures"

    order_id = Column(String(36), ForeignKey("payment_orders.id", ondelete="CASCADE"), nullable=True, index=True)
    authorization_id = Column(String(36), ForeignKey("payment_authorizations.id", ondelete="SET NULL"), nullable=True, index=True)
    provider_capture_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentCaptureStatus), default=PaymentCaptureStatus.PENDING, nullable=False, index=True)

    # Phase 4 Idempotency & Audit
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)
    captured_at = Column(DateTime, nullable=True)

    order = relationship("PaymentOrder", back_populates="captures")
    authorization = relationship("PaymentAuthorization", back_populates="captures")
    refunds = relationship("PaymentRefund", back_populates="capture")


class PaymentRefund(Base):
    __tablename__ = "payment_refunds"

    capture_id = Column(String(36), ForeignKey("payment_captures.id", ondelete="CASCADE"), nullable=False, index=True)
    provider_refund_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    reason = Column(String(255), nullable=True)
    status = Column(SQLEnum(PaymentRefundStatus), default=PaymentRefundStatus.PENDING, nullable=False, index=True)

    # Phase 4 Idempotency & Audit
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)
    refunded_at = Column(DateTime, nullable=True)

    capture = relationship("PaymentCapture", back_populates="refunds")


class WebhookEvent(Base):
    """Durable record of incoming payment webhooks for verification and deduplication."""
    __tablename__ = "webhook_events"

    provider = Column(String(50), nullable=False, default="PAYPAL")
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    resource_type = Column(String(50), nullable=True)
    signature_verification_status = Column(
        SQLEnum(WebhookSignatureStatus),
        default=WebhookSignatureStatus.PENDING,
        nullable=False
    )
    processing_status = Column(
        SQLEnum(WebhookProcessingStatus),
        default=WebhookProcessingStatus.PENDING,
        nullable=False,
        index=True
    )
    payload = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    failure_reason = Column(Text, nullable=True)
    processed_at = Column(DateTime, nullable=True)


class PayoutRecipient(Base):
    """
    Verified payout destination for an organization.
    Stores destination details encrypted at rest using AES-GCM / Fernet, with a SHA-256 hash for deduplication.
    """
    __tablename__ = "payout_recipients"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    recipient_type = Column(SQLEnum(RecipientType), default=RecipientType.PAYPAL_EMAIL, nullable=False)
    payout_method = Column(SQLEnum(DisbursementMethod), default=DisbursementMethod.PAYPAL_PAYOUTS, nullable=False)
    encrypted_destination = Column(Text, nullable=False)
    destination_masked = Column(String(128), nullable=False)
    destination_hash = Column(String(64), nullable=False, index=True)
    status = Column(SQLEnum(PayoutRecipientStatus), default=PayoutRecipientStatus.PENDING_VERIFICATION, nullable=False, index=True)
    verification_notes = Column(Text, nullable=True)
    verified_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    is_primary = Column(Boolean, default=False, nullable=False)

    disbursements = relationship("Disbursement", back_populates="recipient")


class PayoutBatch(Base):
    """
    Batched outbound payout submission sent to payment provider (e.g. PayPal Payouts).
    Groups one or more approved milestone disbursements.
    """
    __tablename__ = "payout_batches"

    batch_reference_id = Column(String(255), unique=True, nullable=False, index=True)
    provider = Column(String(50), default="PAYPAL_PAYOUTS", nullable=False)
    provider_batch_id = Column(String(255), unique=True, nullable=True, index=True)
    status = Column(SQLEnum(DisbursementStatus), default=DisbursementStatus.REQUESTED, nullable=False, index=True)
    currency = Column(String(3), default="USD", nullable=False)
    total_items_count = Column(Integer, default=0, nullable=False)
    total_amount_minor_units = Column(Integer, default=0, nullable=False)
    total_fees_minor_units = Column(Integer, default=0, nullable=False)
    submitted_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    raw_response = Column(Text, nullable=True)

    disbursements = relationship("Disbursement", back_populates="batch")


class Disbursement(Base):
    """
    Milestone-based outbound payout record.
    Subject to strict eligibility, maker-checker financial authorization, and double-entry accounting.
    """
    __tablename__ = "disbursements"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    review_case_id = Column(String(36), ForeignKey("review_cases.id", ondelete="SET NULL"), nullable=True, index=True)
    recipient_id = Column(String(36), ForeignKey("payout_recipients.id", ondelete="SET NULL"), nullable=True, index=True)
    payout_batch_id = Column(String(36), ForeignKey("payout_batches.id", ondelete="SET NULL"), nullable=True, index=True)

    provider = Column(String(50), default="PAYPAL_PAYOUTS", nullable=False)
    provider_batch_id = Column(String(255), nullable=True, index=True)
    provider_item_id = Column(String(255), nullable=True, index=True)
    payout_item_id = Column(String(255), nullable=True)  # Legacy alias
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)

    amount_minor_units = Column(Integer, nullable=False)
    gross_amount_minor_units = Column(Integer, nullable=True)
    provider_fee_minor_units = Column(Integer, default=0, nullable=False)
    net_amount_minor_units = Column(Integer, nullable=True)
    currency = Column(String(3), nullable=False, default="USD")

    status = Column(SQLEnum(DisbursementStatus), default=DisbursementStatus.REQUESTED, nullable=False, index=True)

    requested_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    requested_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    disbursed_at = Column(DateTime, nullable=True)
    failure_reason = Column(Text, nullable=True)

    recipient = relationship("PayoutRecipient", back_populates="disbursements")
    batch = relationship("PayoutBatch", back_populates="disbursements")
    approvals = relationship("PayoutApproval", back_populates="disbursement", cascade="all, delete-orphan")


class PayoutApproval(Base):
    """
    Immutable Maker-Checker authorization log for milestone payouts.
    Records every approval or rejection action taken by authorized finance officers.
    """
    __tablename__ = "payout_approvals"

    disbursement_id = Column(String(36), ForeignKey("disbursements.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(32), nullable=False)
    actor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=False)
    actor_role = Column(String(32), nullable=False)
    notes = Column(Text, nullable=True)
    decided_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    disbursement = relationship("Disbursement", back_populates="approvals")


class ReconciliationRun(Base):
    """
    Automated or manual financial reconciliation session between local ledger and external gateway.
    """
    __tablename__ = "reconciliation_runs"

    run_number = Column(String(64), unique=True, nullable=False, index=True)
    provider = Column(String(50), default="PAYPAL", nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    status = Column(SQLEnum(ReconciliationRunStatus), default=ReconciliationRunStatus.PENDING, nullable=False, index=True)
    total_records_analyzed = Column(Integer, default=0, nullable=False)
    matched_count = Column(Integer, default=0, nullable=False)
    discrepancy_count = Column(Integer, default=0, nullable=False)
    total_local_minor_units = Column(Integer, default=0, nullable=False)
    total_gateway_minor_units = Column(Integer, default=0, nullable=False)
    executed_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    notes = Column(Text, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    items = relationship("ReconciliationItem", back_populates="run", cascade="all, delete-orphan")


class ReconciliationItem(Base):
    """
    Detailed audit item within a reconciliation run comparing local records with gateway records.
    """
    __tablename__ = "reconciliation_items"

    reconciliation_run_id = Column(String(36), ForeignKey("reconciliation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    item_type = Column(String(32), nullable=False)
    reference_id = Column(String(255), nullable=True, index=True)
    provider_transaction_id = Column(String(255), nullable=True, index=True)
    local_amount_minor_units = Column(Integer, default=0, nullable=False)
    gateway_amount_minor_units = Column(Integer, default=0, nullable=False)
    local_status = Column(String(50), nullable=True)
    gateway_status = Column(String(50), nullable=True)
    reconciliation_status = Column(SQLEnum(ReconciliationStatus), default=ReconciliationStatus.MATCHED, nullable=False, index=True)
    discrepancy_type = Column(SQLEnum(DiscrepancyType), nullable=True)
    discrepancy_resolution = Column(SQLEnum(DiscrepancyResolution), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    resolved_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    run = relationship("ReconciliationRun", back_populates="items")


class SettlementRecord(Base):
    """
    External settlement or payout report line ingested from payment gateway.
    """
    __tablename__ = "settlement_records"

    provider = Column(String(50), default="PAYPAL", nullable=False)
    settlement_batch_id = Column(String(255), nullable=True, index=True)
    transaction_id = Column(String(255), unique=True, nullable=False, index=True)
    transaction_type = Column(String(50), nullable=False)
    gross_amount_minor_units = Column(Integer, nullable=False)
    fee_amount_minor_units = Column(Integer, default=0, nullable=False)
    net_amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    settled_at = Column(DateTime, nullable=False)
    raw_data = Column(Text, nullable=True)


class ProcessorFeeRecord(Base):
    """
    Audited payment processor fees incurred during transaction processing.
    """
    __tablename__ = "processor_fee_records"

    provider = Column(String(50), default="PAYPAL", nullable=False)
    transaction_reference_id = Column(String(255), unique=True, nullable=False, index=True)
    fee_type = Column(String(50), default="TRANSACTION", nullable=False)
    fee_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    assessed_at = Column(DateTime, nullable=False)
    journal_entry_id = Column(String(36), ForeignKey("journal_entries.id", ondelete="SET NULL"), nullable=True)
