"""
exceptions.py - Payment Integration Specific Exceptions
=======================================================
Translates raw payment gateway error codes into strongly-typed domain exceptions.
"""

from typing import Optional, Dict, Any


class PaymentException(Exception):
    """Base exception for all payment-related errors."""
    def __init__(self, message: str, code: str = "PAYMENT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class PaymentProviderException(PaymentException):
    """Raised when external payment provider (e.g. PayPal) returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PAYMENT_PROVIDER_ERROR", details=details)
        self.status_code = status_code


class PaymentAuthenticationException(PaymentException):
    """Raised when provider OAuth credentials fail or access token cannot be acquired."""
    def __init__(self, message: str = "Payment provider authentication failed"):
        super().__init__(message, code="PAYMENT_AUTH_ERROR")


class PaymentOrderNotFoundException(PaymentException):
    """Raised when a referenced payment order does not exist at provider or locally."""
    def __init__(self, order_id: str):
        super().__init__(f"Payment order {order_id} not found", code="PAYMENT_ORDER_NOT_FOUND", details={"order_id": order_id})


class PaymentCaptureFailedException(PaymentException):
    """Raised when an order capture attempt is declined or fails validation."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PAYMENT_CAPTURE_FAILED", details=details)


class PaymentRefundFailedException(PaymentException):
    """Raised when a refund attempt is declined or fails eligibility criteria."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="PAYMENT_REFUND_FAILED", details=details)


class WebhookVerificationFailedException(PaymentException):
    """Raised when an incoming webhook cryptographic signature is invalid or forged."""
    def __init__(self, message: str = "Webhook signature verification failed"):
        super().__init__(message, code="WEBHOOK_VERIFICATION_FAILED")


class InvalidPaymentAmountException(PaymentException):
    """Raised when a payment amount violates currency scale or limit constraints."""
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_PAYMENT_AMOUNT")


class CurrencyMismatchException(PaymentException):
    """Raised when payment currency does not match target project or milestone."""
    def __init__(self, expected: str, actual: str):
        super().__init__(
            f"Currency mismatch: expected {expected}, got {actual}",
            code="CURRENCY_MISMATCH",
            details={"expected": expected, "actual": actual}
        )

