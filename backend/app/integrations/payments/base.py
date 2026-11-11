"""
base.py - Abstract Payment Provider Interface
=============================================
Defines the standard contract for external payment processors (e.g. PayPal REST API Orders v2).
Never pretends to perform transactions without real gateway execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from backend.app.core.currency import Money
from backend.app.core.constants import PaymentOrderStatus, PaymentAuthorizationStatus, PaymentCaptureStatus


class PaymentProviderInterface(ABC):
    """Abstract interface decoupling business workflows from specific payment vendors."""

    @abstractmethod
    def create_order(
        self,
        amount: Money,
        reference_id: str,
        return_url: str,
        cancel_url: str,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates an order on the payment processor with intent CAPTURE."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Retrieves current order state from the payment processor."""
        pass

    @abstractmethod
    def capture_order(
        self,
        order_id: str,
        amount: Optional[Money] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Captures an approved PayPal order directly without an intermediate authorization."""
        pass

    @abstractmethod
    def get_capture(self, capture_id: str) -> Dict[str, Any]:
        """Retrieves details of a settled payment capture."""
        pass

    @abstractmethod
    def authorize_payment(self, order_id: str) -> Dict[str, Any]:
        """Places a pre-authorization hold on donor funds."""
        pass

    @abstractmethod
    def capture_authorization(self, authorization_id: str, amount: Optional[Money] = None) -> Dict[str, Any]:
        """Settles authorized funds into the merchant balance upon approved review."""
        pass

    @abstractmethod
    def void_authorization(self, authorization_id: str) -> bool:
        """Releases an authorized hold back to the donor (e.g., if milestone rejected)."""
        pass

    @abstractmethod
    def refund_capture(
        self,
        capture_id: str,
        amount: Money,
        reason: str,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refunds captured funds back to donor."""
        pass

    @abstractmethod
    def get_refund(self, refund_id: str) -> Dict[str, Any]:
        """Retrieves details of a processed refund."""
        pass

    @abstractmethod
    def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        raw_body: bytes,
        webhook_id: Optional[str] = None
    ) -> bool:
        """Verifies cryptographic signature of incoming provider webhooks."""
        pass
