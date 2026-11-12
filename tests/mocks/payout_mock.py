"""
payout_mock.py - High-Fidelity In-Memory Mock Payout Provider
=============================================================
Simulates genuine PayPal Payouts batch processing for testing:
- Validates recipient destinations
- Simulates synchronous and asynchronous batch processing
- Models per-item fees, failure modes, duplicates, and cancellations
- Deterministic behavior without network requests
"""

import uuid
from typing import Dict, Any, Optional, Set
from backend.app.core.constants import RecipientType
from backend.app.integrations.payouts.base import PayoutProviderInterface
from backend.app.integrations.payouts.schemas import (
    PayoutRecipientValidationResult,
    PayoutBatchRequest,
    PayoutBatchResponse,
    PayoutItemResponse
)
from backend.app.integrations.payouts.exceptions import (
    PayoutBatchFailedException,
    PayoutItemFailedException,
    PayoutAuthenticationException,
    PayoutProviderException,
    DuplicatePayoutBatchException
)


class MockPayoutProvider(PayoutProviderInterface):
    """In-memory mock implementation of PayoutProviderInterface."""

    def __init__(self):
        self.batches: Dict[str, Dict[str, Any]] = {}  # payout_batch_id -> data
        self.sender_batch_ids: Set[str] = set()
        self.items: Dict[str, Dict[str, Any]] = {}    # payout_item_id -> data
        self.batch_counter = 0
        self.item_counter = 0

        # Fault injection toggles
        self.simulate_batch_failure: bool = False
        self.simulate_auth_failure: bool = False
        self.simulate_network_error: bool = False
        self.simulate_item_failure_ids: Set[str] = set()
        self.default_item_status: str = "SUCCESS"  # SUCCESS, PENDING, FAILED
        self.default_fee_minor_units: int = 25     # $0.25 standard PayPal fee

    def validate_recipient(
        self,
        recipient_type: RecipientType,
        destination: str
    ) -> PayoutRecipientValidationResult:
        clean = destination.strip()
        if recipient_type == RecipientType.PAYPAL_EMAIL:
            is_valid = "@" in clean and "." in clean and len(clean) >= 5
            return PayoutRecipientValidationResult(
                recipient_type=recipient_type,
                destination=destination,
                is_valid=is_valid,
                normalized_destination=clean.lower(),
                error_message=None if is_valid else "Invalid PayPal email format."
            )
        return PayoutRecipientValidationResult(
            recipient_type=recipient_type,
            destination=destination,
            is_valid=bool(clean),
            normalized_destination=clean,
            error_message=None if clean else "Destination cannot be empty."
        )

    def create_payout_batch(
        self,
        batch_request: PayoutBatchRequest
    ) -> PayoutBatchResponse:
        if self.simulate_auth_failure:
            raise PayoutAuthenticationException("Mock simulated authentication error")
        if self.simulate_network_error:
            raise PayoutProviderException("Mock simulated network timeout", status_code=504)
        if self.simulate_batch_failure:
            raise PayoutBatchFailedException("Mock simulated batch rejection by provider")

        if batch_request.sender_batch_id in self.sender_batch_ids:
            raise DuplicatePayoutBatchException(batch_request.sender_batch_id)

        self.sender_batch_ids.add(batch_request.sender_batch_id)
        self.batch_counter += 1
        payout_batch_id = f"MOCK-BATCH-{self.batch_counter:04d}"

        total_amount = sum(it.amount_minor_units for it in batch_request.items)
        total_fees = len(batch_request.items) * self.default_fee_minor_units
        currency = batch_request.items[0].currency if batch_request.items else "USD"

        item_responses = []
        for it in batch_request.items:
            self.item_counter += 1
            item_id = f"MOCK-ITEM-{self.item_counter:04d}"
            tx_id = f"MOCK-TX-{self.item_counter:04d}"

            is_failed = (
                it.disbursement_id in self.simulate_item_failure_ids
                or it.sender_item_id in self.simulate_item_failure_ids
            )
            item_status = "FAILED" if is_failed else self.default_item_status
            item_fee = self.default_fee_minor_units if item_status == "SUCCESS" else 0

            item_data = {
                "payout_item_id": item_id,
                "sender_item_id": it.sender_item_id,
                "transaction_id": tx_id if item_status == "SUCCESS" else None,
                "transaction_status": item_status,
                "amount_minor_units": it.amount_minor_units,
                "fee_minor_units": item_fee,
                "currency": it.currency,
                "receiver": it.receiver,
                "errors": {"message": "Simulated item failure"} if is_failed else None
            }
            self.items[item_id] = item_data
            item_responses.append(PayoutItemResponse(**item_data))

        batch_status = "SUCCESS" if self.default_item_status == "SUCCESS" else "PENDING"
        batch_record = {
            "payout_batch_id": payout_batch_id,
            "sender_batch_id": batch_request.sender_batch_id,
            "batch_status": batch_status,
            "total_items": len(batch_request.items),
            "total_amount_minor_units": total_amount,
            "total_fee_minor_units": total_fees,
            "currency": currency,
            "items": item_responses,
            "raw_response": {"id": payout_batch_id, "status": batch_status}
        }
        self.batches[payout_batch_id] = batch_record

        return PayoutBatchResponse(**batch_record)

    def get_payout_batch_status(
        self,
        payout_batch_id: str
    ) -> PayoutBatchResponse:
        if payout_batch_id not in self.batches:
            raise PayoutBatchFailedException(f"Payout batch {payout_batch_id} not found in mock store.")
        data = self.batches[payout_batch_id]
        return PayoutBatchResponse(**data)

    def get_payout_item_status(
        self,
        payout_item_id: str
    ) -> PayoutItemResponse:
        if payout_item_id not in self.items:
            raise PayoutItemFailedException(f"Payout item {payout_item_id} not found in mock store.")
        data = self.items[payout_item_id]
        return PayoutItemResponse(**data)

    def cancel_unclaimed_payout_item(
        self,
        payout_item_id: str
    ) -> Dict[str, Any]:
        if payout_item_id not in self.items:
            raise PayoutItemFailedException(f"Payout item {payout_item_id} not found in mock store.")
        self.items[payout_item_id]["transaction_status"] = "CANCELED"
        return {"payout_item_id": payout_item_id, "status": "CANCELED"}

