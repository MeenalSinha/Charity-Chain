"""
paypal_payouts.py - Genuine PayPal Payouts Batch API v1 Integration
===================================================================
Orchestrates outbound mass disbursements to verified NGO payout destinations:
- OAuth 2.0 Bearer token management with thread-safe caching.
- Batch submission via POST /v1/payments/payouts.
- Deduplication via sender_batch_id.
- Real-time batch and item status inspection.
- Currency formatting with integer minor unit precision.
"""

import re
import time
import base64
import threading
from typing import Dict, Any, Optional, List
import requests

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.core.constants import RecipientType
from backend.app.integrations.payouts.base import PayoutProviderInterface
from backend.app.integrations.payouts.schemas import (
    PayoutRecipientValidationResult,
    PayoutBatchRequest,
    PayoutBatchResponse,
    PayoutItemResponse,
    PayoutItemRequest
)
from backend.app.integrations.payouts.exceptions import (
    PayoutException,
    PayoutProviderException,
    PayoutAuthenticationException,
    PayoutBatchFailedException,
    PayoutItemFailedException,
    RecipientIneligibleException,
    DuplicatePayoutBatchException
)

logger = get_logger("integrations.payouts.paypal")

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class PayPalPayoutProvider(PayoutProviderInterface):
    """Production integration with PayPal Payouts API v1."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: Optional[str] = None
    ):
        self.client_id = client_id if client_id is not None else settings.PAYPAL_CLIENT_ID
        self.client_secret = client_secret if client_secret is not None else settings.PAYPAL_CLIENT_SECRET
        self.mode = (mode or getattr(settings, "PAYPAL_MODE", "sandbox")).lower()

        if base_url or getattr(settings, "PAYPAL_API_BASE_URL", None):
            self.base_url = (base_url or settings.PAYPAL_API_BASE_URL).rstrip("/")
        elif self.mode == "live":
            self.base_url = "https://api-m.paypal.com"
        else:
            self.base_url = "https://api-m.sandbox.paypal.com"

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()
        self._session = requests.Session()
        self.connect_timeout = getattr(settings, "PAYPAL_CONNECT_TIMEOUT", 10.0)
        self.read_timeout = getattr(settings, "PAYPAL_READ_TIMEOUT", 30.0)

    # -------------------------------------------------------------------------
    # Internal Helpers & OAuth
    # -------------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Retrieves and caches OAuth token for Payouts."""
        now = time.time()
        if self._token and now < self._token_expires_at:
            return self._token

        with self._token_lock:
            if self._token and time.time() < self._token_expires_at:
                return self._token

            if not self.client_id or not self.client_secret:
                raise PayoutAuthenticationException("PayPal credentials missing for Payouts.")

            auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            token_url = f"{self.base_url}/v1/oauth2/token"
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}

            try:
                response = self._session.post(
                    token_url,
                    headers=headers,
                    data=data,
                    timeout=(self.connect_timeout, self.read_timeout)
                )
            except requests.RequestException as exc:
                logger.error("Failed to acquire PayPal OAuth token for payouts", exc_info=True)
                raise PayoutAuthenticationException(f"Network error during PayPal OAuth: {str(exc)}")

            if response.status_code != 200:
                logger.error("PayPal OAuth failed for payouts", extra={"status": response.status_code, "body": response.text})
                raise PayoutAuthenticationException(f"PayPal OAuth failed with status {response.status_code}")

            try:
                token_data = response.json()
            except Exception:
                raise PayoutAuthenticationException("Invalid JSON in PayPal OAuth response")

            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            if not access_token:
                raise PayoutAuthenticationException("No access_token found in PayPal response")

            self._token = access_token
            buffer = getattr(settings, "OAUTH_TOKEN_CACHE_TTL_SECONDS", 300)
            self._token_expires_at = time.time() + max(60, expires_in - buffer)
            return self._token

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Dispatches an authenticated HTTP request to PayPal API."""
        token = self._get_access_token()
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)

        try:
            response = self._session.request(
                method,
                url,
                json=json_data,
                headers=headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
        except requests.RequestException as exc:
            logger.error(f"PayPal API network failure on {method} {path}", exc_info=True)
            raise PayoutProviderException(f"PayPal connection error: {str(exc)}")

        if response.status_code in (200, 201):
            try:
                return response.json()
            except Exception:
                return {}

        # Error response handling
        try:
            err_data = response.json()
        except Exception:
            err_data = {"error": response.text}

        name = err_data.get("name")
        message = err_data.get("message", response.text)

        if response.status_code == 409 or name == "DUPLICATE_BATCH_ID":
            raise DuplicatePayoutBatchException(json_data.get("sender_batch_header", {}).get("sender_batch_id", "unknown"))

        logger.error(f"PayPal Payouts error {response.status_code} on {path}: {message}", extra={"details": err_data})
        raise PayoutProviderException(f"PayPal Payouts Error ({response.status_code}): {message}", status_code=response.status_code, details=err_data)

    @staticmethod
    def format_money_value(minor_units: int) -> str:
        """Formats integer minor units into standard decimal string without float arithmetic."""
        negative = minor_units < 0
        abs_val = abs(minor_units)
        dollars = abs_val // 100
        cents = abs_val % 100
        sign = "-" if negative else ""
        return f"{sign}{dollars}.{cents:02d}"

    @staticmethod
    def parse_money_value(val_str: str) -> int:
        """Parses decimal string into integer minor units safely."""
        parts = val_str.strip().split(".")
        dollars = int(parts[0])
        cents = int(parts[1][:2].ljust(2, "0")) if len(parts) > 1 else 0
        return dollars * 100 + cents

    # -------------------------------------------------------------------------
    # Public Provider Methods
    # -------------------------------------------------------------------------

    def validate_recipient(
        self,
        recipient_type: RecipientType,
        destination: str
    ) -> PayoutRecipientValidationResult:
        """Validates recipient format (e.g. valid PayPal email)."""
        clean = destination.strip()
        if recipient_type == RecipientType.PAYPAL_EMAIL:
            if not EMAIL_REGEX.match(clean):
                return PayoutRecipientValidationResult(
                    recipient_type=recipient_type,
                    destination=destination,
                    is_valid=False,
                    normalized_destination=clean.lower(),
                    error_message="Invalid PayPal email address format."
                )
            return PayoutRecipientValidationResult(
                recipient_type=recipient_type,
                destination=destination,
                is_valid=True,
                normalized_destination=clean.lower(),
                error_message=None
            )

        # For bank accounts or UPI:
        if not clean:
            return PayoutRecipientValidationResult(
                recipient_type=recipient_type,
                destination=destination,
                is_valid=False,
                normalized_destination=clean,
                error_message="Destination identifier cannot be empty."
            )

        return PayoutRecipientValidationResult(
            recipient_type=recipient_type,
            destination=destination,
            is_valid=True,
            normalized_destination=clean,
            error_message=None
        )

    def create_payout_batch(
        self,
        batch_request: PayoutBatchRequest
    ) -> PayoutBatchResponse:
        """Submits batch of payout items to PayPal via POST /v1/payments/payouts."""
        items_payload = []
        total_minor = 0
        currency = batch_request.items[0].currency if batch_request.items else "USD"

        for item in batch_request.items:
            total_minor += item.amount_minor_units
            items_payload.append({
                "recipient_type": "EMAIL" if item.recipient_type == RecipientType.PAYPAL_EMAIL else "EMAIL",
                "amount": {
                    "value": self.format_money_value(item.amount_minor_units),
                    "currency": item.currency
                },
                "note": item.note or f"Milestone disbursement: {item.disbursement_id}",
                "sender_item_id": item.sender_item_id,
                "receiver": item.receiver
            })

        payload = {
            "sender_batch_header": {
                "sender_batch_id": batch_request.sender_batch_id,
                "email_subject": batch_request.email_subject,
                "email_message": batch_request.email_message or "Thank you for creating verifiable impact with CharityChain."
            },
            "items": items_payload
        }

        res = self._request("POST", "/v1/payments/payouts", json_data=payload)
        batch_header = res.get("batch_header", {})
        payout_batch_id = batch_header.get("payout_batch_id", "")
        batch_status = batch_header.get("batch_status", "PENDING")

        fees_obj = batch_header.get("fees", {})
        fee_minor = self.parse_money_value(fees_obj.get("value", "0.00")) if fees_obj else 0

        # Construct initial response items
        response_items = []
        for item in batch_request.items:
            response_items.append(
                PayoutItemResponse(
                    payout_item_id=f"ITEM-{item.sender_item_id}",
                    sender_item_id=item.sender_item_id,
                    transaction_id=None,
                    transaction_status=batch_status,
                    amount_minor_units=item.amount_minor_units,
                    fee_minor_units=0,
                    currency=item.currency,
                    receiver=item.receiver
                )
            )

        return PayoutBatchResponse(
            payout_batch_id=payout_batch_id,
            sender_batch_id=batch_request.sender_batch_id,
            batch_status=batch_status,
            total_items=len(batch_request.items),
            total_amount_minor_units=total_minor,
            total_fee_minor_units=fee_minor,
            currency=currency,
            items=response_items,
            raw_response=res
        )

    def get_payout_batch_status(
        self,
        payout_batch_id: str
    ) -> PayoutBatchResponse:
        """Queries status of an existing batch: GET /v1/payments/payouts/{id}."""
        res = self._request("GET", f"/v1/payments/payouts/{payout_batch_id}")
        batch_header = res.get("batch_header", {})
        sender_batch_header = batch_header.get("sender_batch_header", {})
        sender_batch_id = sender_batch_header.get("sender_batch_id", "")
        batch_status = batch_header.get("batch_status", "PENDING")

        amount_obj = batch_header.get("amount", {})
        total_minor = self.parse_money_value(amount_obj.get("value", "0.00")) if amount_obj else 0
        fees_obj = batch_header.get("fees", {})
        total_fee = self.parse_money_value(fees_obj.get("value", "0.00")) if fees_obj else 0
        currency = amount_obj.get("currency", "USD") if amount_obj else "USD"

        items = []
        for it in res.get("items", []):
            payout_item_id = it.get("payout_item_id", "")
            item_status = it.get("transaction_status", "PENDING")
            item_amount_obj = it.get("payout_item", {}).get("amount", {})
            amt_minor = self.parse_money_value(item_amount_obj.get("value", "0.00")) if item_amount_obj else 0
            item_fee_obj = it.get("payout_item_fee", {})
            fee_minor = self.parse_money_value(item_fee_obj.get("value", "0.00")) if item_fee_obj else 0
            receiver = it.get("payout_item", {}).get("receiver", "")
            sender_item_id = it.get("payout_item", {}).get("sender_item_id", "")
            tx_id = it.get("transaction_id")
            errors = it.get("errors")

            items.append(
                PayoutItemResponse(
                    payout_item_id=payout_item_id,
                    sender_item_id=sender_item_id,
                    transaction_id=tx_id,
                    transaction_status=item_status,
                    amount_minor_units=amt_minor,
                    fee_minor_units=fee_minor,
                    currency=item_amount_obj.get("currency", currency),
                    receiver=receiver,
                    errors=errors
                )
            )

        return PayoutBatchResponse(
            payout_batch_id=payout_batch_id,
            sender_batch_id=sender_batch_id,
            batch_status=batch_status,
            total_items=len(items),
            total_amount_minor_units=total_minor,
            total_fee_minor_units=total_fee,
            currency=currency,
            items=items,
            raw_response=res
        )

    def get_payout_item_status(
        self,
        payout_item_id: str
    ) -> PayoutItemResponse:
        """Queries status of an individual item: GET /v1/payments/payouts-item/{id}."""
        res = self._request("GET", f"/v1/payments/payouts-item/{payout_item_id}")
        item_status = res.get("transaction_status", "PENDING")
        item_amount_obj = res.get("payout_item", {}).get("amount", {})
        amt_minor = self.parse_money_value(item_amount_obj.get("value", "0.00")) if item_amount_obj else 0
        item_fee_obj = res.get("payout_item_fee", {})
        fee_minor = self.parse_money_value(item_fee_obj.get("value", "0.00")) if item_fee_obj else 0
        receiver = res.get("payout_item", {}).get("receiver", "")
        sender_item_id = res.get("payout_item", {}).get("sender_item_id", "")
        tx_id = res.get("transaction_id")
        errors = res.get("errors")

        return PayoutItemResponse(
            payout_item_id=payout_item_id,
            sender_item_id=sender_item_id,
            transaction_id=tx_id,
            transaction_status=item_status,
            amount_minor_units=amt_minor,
            fee_minor_units=fee_minor,
            currency=item_amount_obj.get("currency", "USD"),
            receiver=receiver,
            errors=errors
        )

    def cancel_unclaimed_payout_item(
        self,
        payout_item_id: str
    ) -> Dict[str, Any]:
        """Cancels an unclaimed payout item: POST /v1/payments/payouts-item/{id}/cancel."""
        return self._request("POST", f"/v1/payments/payouts-item/{payout_item_id}/cancel")

    def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        raw_body: bytes,
        webhook_id: Optional[str] = None
    ) -> bool:
        """
        Cryptographically validates incoming PayPal payout webhook signatures via PayPal's
        /v1/notifications/verify-webhook-signature endpoint.
        """
        import json
        normalized_headers = {k.lower(): v for k, v in headers.items()}
        auth_algo = normalized_headers.get("paypal-auth-algo")
        cert_url = normalized_headers.get("paypal-cert-url")
        transmission_id = normalized_headers.get("paypal-transmission-id")
        transmission_sig = normalized_headers.get("paypal-transmission-sig")
        transmission_time = normalized_headers.get("paypal-transmission-time")
        target_webhook_id = webhook_id or getattr(settings, "PAYPAL_PAYOUTS_WEBHOOK_ID", None) or getattr(settings, "PAYPAL_WEBHOOK_ID", None)

        if not all([auth_algo, cert_url, transmission_id, transmission_sig, transmission_time, target_webhook_id]):
            logger.warning("Missing required PayPal payout webhook verification headers or webhook_id")
            return False

        try:
            webhook_event = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            logger.error("Failed to parse raw payout webhook body as JSON for signature verification")
            return False

        verification_payload = {
            "auth_algo": auth_algo,
            "cert_url": cert_url,
            "transmission_id": transmission_id,
            "transmission_sig": transmission_sig,
            "transmission_time": transmission_time,
            "webhook_id": target_webhook_id,
            "webhook_event": webhook_event
        }

        try:
            res = self._request("POST", "/v1/notifications/verify-webhook-signature", json_data=verification_payload)
            status = res.get("verification_status")
            if status == "SUCCESS":
                return True
            logger.warning(f"PayPal payout webhook signature verification rejected with status: {status}")
            return False
        except Exception as exc:
            logger.error(f"Error calling PayPal payout webhook signature verification: {str(exc)}")
            return False

