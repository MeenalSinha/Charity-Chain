"""
capability.py - Outbound Payout Corridor & Regulatory Capabilities
==================================================================
Defines supported currencies, recipient corridors, regulatory limitations,
and per-corridor transaction thresholds.

Regulatory Note on India (IN):
-----------------------------
PayPal India accounts cannot receive domestic personal/commercial payouts via the
standard mass Payouts API without RBI/FEMA export compliance and Purpose Codes.
CharityChain enforces international cross-border compliance:
Indian NGOs receiving international disbursements must provide export-compliant documentation
or configure domestic direct bank rail / UPI settlement when operating domestically.
"""

from typing import Dict, Tuple, Optional
from backend.app.core.constants import RecipientType, DisbursementMethod


SUPPORTED_PAYOUT_CURRENCIES = {
    "USD": {"min_minor_units": 100, "max_item_minor_units": 2000000, "max_batch_minor_units": 100000000},
    "EUR": {"min_minor_units": 100, "max_item_minor_units": 1800000, "max_batch_minor_units": 90000000},
    "GBP": {"min_minor_units": 100, "max_item_minor_units": 1600000, "max_batch_minor_units": 80000000},
    "CAD": {"min_minor_units": 100, "max_item_minor_units": 2500000, "max_batch_minor_units": 125000000},
    "AUD": {"min_minor_units": 100, "max_item_minor_units": 2500000, "max_batch_minor_units": 125000000},
    "INR": {"min_minor_units": 10000, "max_item_minor_units": 50000000, "max_batch_minor_units": 2500000000},
}


class PayoutCapabilityMatrix:
    """Evaluates regulatory feasibility and limits for outbound disbursement requests."""

    @classmethod
    def is_currency_supported(cls, currency: str) -> bool:
        return currency.upper() in SUPPORTED_PAYOUT_CURRENCIES

    @classmethod
    def get_corridor_limits(cls, currency: str) -> Dict[str, int]:
        curr = currency.upper()
        if curr not in SUPPORTED_PAYOUT_CURRENCIES:
            return {"min_minor_units": 100, "max_item_minor_units": 1000000, "max_batch_minor_units": 50000000}
        return SUPPORTED_PAYOUT_CURRENCIES[curr]

    @classmethod
    def validate_recipient_corridor(
        cls,
        country_code: str,
        currency: str,
        method: DisbursementMethod
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates country corridor regulatory constraints.
        Special handling for India (IN) domestic restrictions.
        """
        c_code = country_code.upper()
        curr = currency.upper()

        if c_code == "IN":
            if method == DisbursementMethod.PAYPAL_PAYOUTS and curr == "INR":
                return (
                    False,
                    "Domestic INR payouts via PayPal Payouts are restricted under RBI regulations. "
                    "Use international currency (USD/EUR/GBP) for foreign donations or bank transfer for domestic INR."
                )

        return True, None

