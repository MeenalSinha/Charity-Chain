"""
schemas.py - Data Contracts for Outbound Payout Operations
==========================================================
Defines request and response structures for payout batches, individual payout items,
recipient destination validation, and provider status reports.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from backend.app.core.constants import RecipientType, DisbursementStatus, DisbursementMethod


class PayoutRecipientValidationResult(BaseModel):
    """Result of validating a recipient's destination format and integrity."""
    recipient_type: RecipientType
    destination: str
    is_valid: bool
    normalized_destination: str
    error_message: Optional[str] = None


class PayoutItemRequest(BaseModel):
    """Specification of an individual disbursement item within a batch."""
    disbursement_id: str = Field(..., description="Internal CharityChain disbursement UUID")
    recipient_type: RecipientType = Field(default=RecipientType.PAYPAL_EMAIL)
    receiver: str = Field(..., description="Recipient destination, e.g. verified PayPal email")
    amount_minor_units: int = Field(..., gt=0, description="Gross payout amount in integer minor units (cents)")
    currency: str = Field(default="USD", description="3-letter ISO-4217 currency code")
    note: Optional[str] = Field(default=None, description="Note delivered to payout recipient")
    sender_item_id: str = Field(..., description="Unique client item identifier for idempotency")


class PayoutItemResponse(BaseModel):
    """Provider response details for an individual payout item."""
    payout_item_id: str
    sender_item_id: str
    transaction_id: Optional[str] = None
    transaction_status: str  # SUCCESS, PENDING, FAILED, UNCLAIMED, RETURNED, BLOCKED
    amount_minor_units: int
    fee_minor_units: int = 0
    currency: str = "USD"
    receiver: str
    errors: Optional[Dict[str, Any]] = None


class PayoutBatchRequest(BaseModel):
    """Batch submission request to outbound payout provider."""
    sender_batch_id: str = Field(..., description="Unique client-generated batch identifier for idempotency")
    email_subject: str = Field(default="You have a milestone disbursement from CharityChain")
    email_message: Optional[str] = Field(default=None)
    items: List[PayoutItemRequest] = Field(..., min_length=1)


class PayoutBatchResponse(BaseModel):
    """Aggregate provider response for a submitted payout batch."""
    payout_batch_id: str
    sender_batch_id: str
    batch_status: str  # PENDING, PROCESSING, SUCCESS, DENIED, CANCELED
    total_items: int
    total_amount_minor_units: int
    total_fee_minor_units: int = 0
    currency: str = "USD"
    items: List[PayoutItemResponse] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)

