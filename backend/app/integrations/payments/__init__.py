"""
__init__.py - Payment Integration Module
"""

from backend.app.integrations.payments.base import PaymentProviderInterface
from backend.app.integrations.payments.paypal import PayPalPaymentProvider
from backend.app.integrations.payments.factory import get_payment_provider, set_payment_provider
from backend.app.integrations.payments.exceptions import (
    PaymentException,
    PaymentProviderException,
    PaymentAuthenticationException,
    PaymentOrderNotFoundException,
    PaymentCaptureFailedException,
    PaymentRefundFailedException,
    WebhookVerificationFailedException,
    InvalidPaymentAmountException,
    CurrencyMismatchException
)

__all__ = [
    "PaymentProviderInterface",
    "PayPalPaymentProvider",
    "get_payment_provider",
    "set_payment_provider",
    "PaymentException",
    "PaymentProviderException",
    "PaymentAuthenticationException",
    "PaymentOrderNotFoundException",
    "PaymentCaptureFailedException",
    "PaymentRefundFailedException",
    "WebhookVerificationFailedException",
    "InvalidPaymentAmountException",
    "CurrencyMismatchException"
]

