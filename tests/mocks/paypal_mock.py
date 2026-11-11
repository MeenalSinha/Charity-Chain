"""
paypal_mock.py - Deterministic Mock PayPal Payment Provider
============================================================
Simulates genuine PayPal Orders v2 and Payments v2 responses for test execution.
Supports configurable failure modes, order states, and signature checks.
"""

from typing import Dict, Any, Optional
from decimal import Decimal
from backend.app.core.currency import Money, CURRENCY_MINOR_UNITS
from backend.app.integrations.payments.base import PaymentProviderInterface
from backend.app.integrations.payments.exceptions import (
    PaymentOrderNotFoundException, PaymentCaptureFailedException,
    PaymentRefundFailedException, PaymentProviderException
)


class MockPayPalProvider(PaymentProviderInterface):
    """Deterministic, in-memory mock for PayPal Orders and Payments API."""

    def __init__(self):
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.captures: Dict[str, Dict[str, Any]] = {}
        self.refunds: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 1000
        self.capture_counter = 2000
        self.refund_counter = 3000

        # Configurable failure simulations
        self.simulate_order_creation_error = False
        self.simulate_capture_error = False
        self.simulate_refund_error = False
        self.simulate_invalid_webhook = False
        self.simulate_unapproved_order = False

    def format_money_value(self, amount: Money) -> str:
        scale = CURRENCY_MINOR_UNITS.get(amount.currency, 2)
        multiplier = Decimal(10 ** scale)
        major_decimal = Decimal(amount.minor_units) / multiplier
        return f"{major_decimal:.{scale}f}"

    def create_order(
        self,
        amount: Money,
        reference_id: str,
        return_url: str,
        cancel_url: str,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.simulate_order_creation_error:
            raise PaymentProviderException("Simulated order creation error at PayPal", status_code=500)

        self.order_counter += 1
        order_id = f"MOCK-PAYPAL-ORD-{self.order_counter}"
        formatted = self.format_money_value(amount)
        initial_status = "CREATED" if self.simulate_unapproved_order else "APPROVED"

        order_data = {
            "id": order_id,
            "status": initial_status,
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": reference_id,
                    "custom_id": custom_id,
                    "amount": {
                        "currency_code": amount.currency,
                        "value": formatted
                    },
                    "payments": {
                        "captures": []
                    }
                }
            ],
            "links": [
                {"rel": "approve", "href": f"https://www.sandbox.paypal.com/checkoutnow?token={order_id}"},
                {"rel": "self", "href": f"https://api-m.sandbox.paypal.com/v2/checkout/orders/{order_id}"}
            ]
        }
        self.orders[order_id] = order_data

        return {
            "id": order_id,
            "provider_order_id": order_id,
            "status": initial_status,
            "approval_url": f"https://www.sandbox.paypal.com/checkoutnow?token={order_id}",
            "raw": order_data
        }

    def get_order(self, order_id: str) -> Dict[str, Any]:
        if order_id not in self.orders:
            raise PaymentOrderNotFoundException(f"Order {order_id} not found in mock store")
        return self.orders[order_id]

    def capture_order(
        self,
        order_id: str,
        amount: Optional[Money] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.simulate_capture_error:
            raise PaymentCaptureFailedException("Simulated capture decline at PayPal")

        order = self.orders.get(order_id)
        if not order:
            raise PaymentOrderNotFoundException(f"Order {order_id} not found")

        self.capture_counter += 1
        capture_id = f"MOCK-CAP-{self.capture_counter}"
        pu = order["purchase_units"][0]
        val = pu["amount"]["value"]
        curr = pu["amount"]["currency_code"]

        capture_data = {
            "id": capture_id,
            "status": "COMPLETED",
            "amount": {"value": val, "currency_code": curr},
            "idempotency_key": idempotency_key
        }
        self.captures[capture_id] = capture_data

        # Update order in store
        order["status"] = "COMPLETED"
        pu["payments"]["captures"].append(capture_data)

        return {
            "order_id": order_id,
            "capture_id": capture_id,
            "id": capture_id,
            "status": "COMPLETED",
            "raw": capture_data
        }

    def get_capture(self, capture_id: str) -> Dict[str, Any]:
        if capture_id not in self.captures:
            raise PaymentCaptureFailedException(f"Capture {capture_id} not found in mock store")
        return self.captures[capture_id]

    def refund_capture(
        self,
        capture_id: str,
        amount: Money,
        reason: str,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.simulate_refund_error:
            raise PaymentRefundFailedException("Simulated refund failure at PayPal")

        self.refund_counter += 1
        refund_id = f"MOCK-REF-{self.refund_counter}"
        formatted = self.format_money_value(amount)

        refund_data = {
            "id": refund_id,
            "status": "COMPLETED",
            "amount": {"value": formatted, "currency_code": amount.currency},
            "note_to_payer": reason,
            "idempotency_key": idempotency_key
        }
        self.refunds[refund_id] = refund_data

        return {
            "refund_id": refund_id,
            "id": refund_id,
            "status": "COMPLETED",
            "raw": refund_data
        }

    def get_refund(self, refund_id: str) -> Dict[str, Any]:
        if refund_id not in self.refunds:
            raise PaymentRefundFailedException(f"Refund {refund_id} not found")
        return self.refunds[refund_id]

    def authorize_payment(self, order_id: str) -> Dict[str, Any]:
        return {"status": "CREATED", "id": f"AUTH-{order_id}"}

    def capture_authorization(self, authorization_id: str, amount: Optional[Money] = None) -> Dict[str, Any]:
        return {"status": "COMPLETED", "id": f"CAP-{authorization_id}"}

    def void_authorization(self, authorization_id: str) -> bool:
        return True

    def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        raw_body: bytes,
        webhook_id: Optional[str] = None
    ) -> bool:
        if self.simulate_invalid_webhook:
            return False
        # If transmission-sig is 'INVALID_SIG', return False
        sig = headers.get("paypal-transmission-sig") or headers.get("PAYPAL-TRANSMISSION-SIG")
        if sig == "INVALID_SIG":
            return False
        return True
