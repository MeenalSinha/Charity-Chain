"""
exceptions.py - Strongly-Typed Outbound Payout Domain Exceptions
=================================================================
Maps low-level payment gateway error codes into platform-level domain exceptions.
"""

from typing import Optional, Dict, Any


class PayoutException(Exception):
    """Base exception for all outbound payout and disbursement failures."""
    def __init__(self, message: str, code: str = "PAYOUT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class PayoutProviderException(PayoutException):
    """Raised when the external payout provider returns an HTTP or operational error."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PAYOUT_PROVIDER_ERROR", details=details)
        self.status_code = status_code


class PayoutAuthenticationException(PayoutException):
    """Raised when payout OAuth credentials fail or access token cannot be acquired."""
    def __init__(self, message: str = "Payout provider authentication failed"):
        super().__init__(message, code="PAYOUT_AUTH_ERROR")


class PayoutBatchFailedException(PayoutException):
    """Raised when a payout batch submission is rejected or cannot be processed."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PAYOUT_BATCH_FAILED", details=details)


class PayoutItemFailedException(PayoutException):
    """Raised when an individual payout item within a batch fails."""
    def __init__(self, message: str, item_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        merged = {"item_id": item_id} if item_id else {}
        if details:
            merged.update(details)
        super().__init__(message, code="PAYOUT_ITEM_FAILED", details=merged)


class RecipientIneligibleException(PayoutException):
    """Raised when an NGO recipient destination fails validation or compliance checks."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="RECIPIENT_INELIGIBLE", details=details)


class DuplicatePayoutBatchException(PayoutException):
    """Raised when attempting to submit a payout batch with a duplicate sender_batch_id."""
    def __init__(self, batch_id: str):
        super().__init__(
            f"Payout batch with sender_batch_id '{batch_id}' has already been submitted.",
            code="DUPLICATE_PAYOUT_BATCH",
            details={"sender_batch_id": batch_id}
        )


class PayoutCurrencyUnsupportedException(PayoutException):
    """Raised when payout currency or corridor is not supported by provider."""
    def __init__(self, currency: str):
        super().__init__(
            f"Payout currency '{currency}' is not supported for outbound disbursement.",
            code="UNSUPPORTED_PAYOUT_CURRENCY",
            details={"currency": currency}
        )
