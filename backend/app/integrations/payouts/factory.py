"""
factory.py - Payout Provider Factory & Dependency Injection
===========================================================
Instantiates the configured PayoutProviderInterface implementation.
Allows seamless injection of in-memory test mocks during automated tests.
"""

from typing import Optional
from backend.app.core.config import settings
from backend.app.integrations.payouts.base import PayoutProviderInterface
from backend.app.integrations.payouts.paypal_payouts import PayPalPayoutProvider

_override_provider: Optional[PayoutProviderInterface] = None


def set_payout_provider_override(provider: Optional[PayoutProviderInterface]) -> None:
    """Sets a global override payout provider (used extensively in unit/integration test suites)."""
    global _override_provider
    _override_provider = provider


def get_payout_provider() -> PayoutProviderInterface:
    """Returns the active payout provider instance."""
    if _override_provider is not None:
        return _override_provider

    if getattr(settings, "PAYOUT_PROVIDER", "paypal") == "mock":
        from tests.mocks.payout_mock import MockPayoutProvider
        return MockPayoutProvider()

    return PayPalPayoutProvider()
