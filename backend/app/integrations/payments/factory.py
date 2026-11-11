"""
factory.py - Payment Provider Factory
=====================================
Instantiates the configured PaymentProviderInterface instance.
"""

from typing import Optional
from backend.app.core.config import settings
from backend.app.integrations.payments.base import PaymentProviderInterface
from backend.app.integrations.payments.paypal import PayPalPaymentProvider

_provider_instance: Optional[PaymentProviderInterface] = None


def get_payment_provider() -> PaymentProviderInterface:
    """Returns the configured payment provider singleton."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = PayPalPaymentProvider()
    return _provider_instance


def set_payment_provider(provider: PaymentProviderInterface) -> None:
    """Overrides the active payment provider (useful for testing and mocks)."""
    global _provider_instance
    _provider_instance = provider

