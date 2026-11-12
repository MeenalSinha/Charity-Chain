"""
base.py - Abstract Interface for Outbound Payout Providers
==========================================================
Specifies the contract that all outbound disbursement providers must fulfill.
Enables transparent swapping between genuine PayPal Payouts, bank transfer rails,
and deterministic in-memory test mocks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from backend.app.core.constants import RecipientType
from backend.app.integrations.payouts.schemas import (
    PayoutRecipientValidationResult,
    PayoutBatchRequest,
    PayoutBatchResponse,
    PayoutItemResponse
)


class PayoutProviderInterface(ABC):
    """Abstract Base Class for outbound disbursement integrations."""

    @abstractmethod
    def validate_recipient(
        self,
        recipient_type: RecipientType,
        destination: str
    ) -> PayoutRecipientValidationResult:
        """Validates recipient destination address/format before persisting or initiating payouts."""
        pass

    @abstractmethod
    def create_payout_batch(
        self,
        batch_request: PayoutBatchRequest
    ) -> PayoutBatchResponse:
        """Submits an atomic or multi-item payout batch to the payment provider."""
        pass

    @abstractmethod
    def get_payout_batch_status(
        self,
        payout_batch_id: str
    ) -> PayoutBatchResponse:
        """Queries the current processing status of an entire payout batch."""
        pass

    @abstractmethod
    def get_payout_item_status(
        self,
        payout_item_id: str
    ) -> PayoutItemResponse:
        """Queries the status and fee details of an individual payout item."""
        pass

    @abstractmethod
    def cancel_unclaimed_payout_item(
        self,
        payout_item_id: str
    ) -> Dict[str, Any]:
        """Cancels an unclaimed payout item and returns funds to sender balance."""
        pass

