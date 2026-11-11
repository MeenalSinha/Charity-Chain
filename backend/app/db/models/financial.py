"""
financial.py - Currency-Safe Financial Domain Models
====================================================
Maintains strict separation between:
1. Donated Intent (Donation)
2. Payment Orders (PaymentOrder)
3. Pre-Authorization Holds (PaymentAuthorization)
4. Payment Captures (PaymentCapture)
5. Reversals/Refunds (PaymentRefund)
6. Actual NGO Payouts (Disbursement)
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from backend.app.db.base import Base
from backend.app.core.constants import (
    PaymentOrderStatus, PaymentAuthorizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus
)


class Donation(Base):
    __tablename__ = "donations"

    donor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    impact_tokens_earned = Column(Integer, default=0, nullable=False)

    donor = relationship("User", back_populates="donations")
    milestone = relationship("Milestone", back_populates="donations")
    orders = relationship("PaymentOrder", back_populates="donation")
    certificate = relationship("ImpactCertificate", back_populates="donation", uselist=False)


class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    donation_id = Column(String(36), ForeignKey("donations.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False, default="PAYPAL")
    provider_order_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentOrderStatus), default=PaymentOrderStatus.CREATED, nullable=False)

    donation = relationship("Donation", back_populates="orders")
    authorizations = relationship("PaymentAuthorization", back_populates="order")


class PaymentAuthorization(Base):
    """Represents a temporary card hold. NOT an escrow account."""
    __tablename__ = "payment_authorizations"

    order_id = Column(String(36), ForeignKey("payment_orders.id", ondelete="CASCADE"), nullable=False)
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

    authorization_id = Column(String(36), ForeignKey("payment_authorizations.id", ondelete="CASCADE"), nullable=False)
    provider_capture_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(PaymentCaptureStatus), default=PaymentCaptureStatus.PENDING, nullable=False)

    authorization = relationship("PaymentAuthorization", back_populates="captures")
    refunds = relationship("PaymentRefund", back_populates="capture")


class PaymentRefund(Base):
    __tablename__ = "payment_refunds"

    capture_id = Column(String(36), ForeignKey("payment_captures.id", ondelete="CASCADE"), nullable=False)
    provider_refund_id = Column(String(255), unique=True, nullable=True, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    reason = Column(String(255), nullable=True)
    status = Column(SQLEnum(PaymentRefundStatus), default=PaymentRefundStatus.PENDING, nullable=False)

    capture = relationship("PaymentCapture", back_populates="refunds")


class Disbursement(Base):
    """Direct payout to NGO bank account/PayPal after milestone approval."""
    __tablename__ = "disbursements"

    milestone_id = Column(String(36), ForeignKey("milestones.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    amount_minor_units = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(SQLEnum(DisbursementStatus), default=DisbursementStatus.SCHEDULED, nullable=False)
    payout_batch_id = Column(String(255), nullable=True)
    payout_item_id = Column(String(255), nullable=True)
    disbursed_at = Column(DateTime, nullable=True)

