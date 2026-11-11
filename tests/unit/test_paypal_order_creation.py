"""
test_paypal_order_creation.py - PayPal Order Payload & Formatting Unit Tests
=============================================================================
Tests currency decimal scaling, purchase unit formatting, and capture response extraction.
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.app.core.currency import Money
from backend.app.integrations.payments.paypal import PayPalPaymentProvider
from backend.app.integrations.payments.exceptions import InvalidPaymentAmountException


def test_money_formatting_scales():
    """Verifies that currency minor units are converted into exact strings for PayPal API."""
    provider = PayPalPaymentProvider()

    # USD has scale 2: 2550 cents -> "25.50"
    m_usd = Money(2550, "USD")
    assert provider.format_money_value(m_usd) == "25.50"

    # EUR has scale 2: 10000 cents -> "100.00"
    m_eur = Money(10000, "EUR")
    assert provider.format_money_value(m_eur) == "100.00"

    # JPY has scale 0: 5000 yen -> "5000"
    m_jpy = Money(5000, "JPY")
    assert provider.format_money_value(m_jpy) == "5000"

    # INR has scale 2: 75000 paise -> "750.00"
    m_inr = Money(75000, "INR")
    assert provider.format_money_value(m_inr) == "750.00"


def test_create_order_zero_amount_rejected():
    """Verifies that non-positive monetary values are rejected before calling PayPal API."""
    provider = PayPalPaymentProvider()
    zero_money = Money(0, "USD")

    with pytest.raises(InvalidPaymentAmountException):
        provider.create_order(
            amount=zero_money,
            reference_id="REF-1",
            return_url="http://return",
            cancel_url="http://cancel"
        )


def test_create_order_payload_construction():
    """Verifies that the generated payload includes intent CAPTURE, proper currency, and links."""
    provider = PayPalPaymentProvider(client_id="c_id", client_secret="c_sec")

    mock_resp = {
        "id": "ORD-123456789",
        "status": "CREATED",
        "links": [
            {"rel": "approve", "href": "https://www.sandbox.paypal.com/checkoutnow?token=ORD-123456789"},
            {"rel": "self", "href": "https://api-m.sandbox.paypal.com/v2/checkout/orders/ORD-123456789"}
        ]
    }

    with patch.object(provider, "_get_access_token", return_value="TOKEN_ABC"):
        with patch.object(provider, "_request", return_value=mock_resp) as mock_req:
            res = provider.create_order(
                amount=Money(5000, "USD"),
                reference_id="DON-DONATION-01",
                return_url="http://localhost:8501/?page=donation_result",
                cancel_url="http://localhost:8501/?page=donation_cancel",
                custom_id="DON-DONATION-01"
            )

            assert res["id"] == "ORD-123456789"
            assert res["status"] == "CREATED"
            assert "token=ORD-123456789" in res["approval_url"]
            mock_req.assert_called_once()
            call_args = mock_req.call_args[1]
            assert call_args["json_data"]["intent"] == "CAPTURE"
            assert call_args["json_data"]["purchase_units"][0]["amount"]["value"] == "50.00"
            assert call_args["json_data"]["purchase_units"][0]["amount"]["currency_code"] == "USD"
