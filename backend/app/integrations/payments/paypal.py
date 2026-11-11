"""
paypal.py - Official PayPal REST API Orders v2 Integration
==========================================================
Implements high-assurance integration with the PayPal Orders v2 and Payments v2 APIs.
Supports:
1. OAuth 2.0 token acquisition and caching
2. Orders API v2 create, get, capture
3. Payments API v2 capture details, refunds, refund details
4. Notifications API v1 webhook signature verification
5. Exponential retry policies on transient network errors
6. Structured logging and error translation
"""

import base64
import json
import time
import threading
from decimal import Decimal
from typing import Dict, Any, Optional
import requests

from backend.app.core.config import settings
from backend.app.core.currency import Money, CURRENCY_MINOR_UNITS
from backend.app.core.logging import get_logger
from backend.app.integrations.payments.base import PaymentProviderInterface
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

logger = get_logger("integrations.payments.paypal")


class PayPalPaymentProvider(PaymentProviderInterface):
    """Production-grade PayPal REST API v2 client."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
        mode: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.client_id = client_id if client_id is not None else settings.PAYPAL_CLIENT_ID
        self.client_secret = client_secret if client_secret is not None else settings.PAYPAL_CLIENT_SECRET
        self.webhook_id = webhook_id if webhook_id is not None else settings.PAYPAL_WEBHOOK_ID
        self.mode = (mode or settings.PAYPAL_MODE).lower()

        if base_url or settings.PAYPAL_API_BASE_URL:
            self.base_url = (base_url or settings.PAYPAL_API_BASE_URL).rstrip("/")
        elif self.mode == "live":
            self.base_url = "https://api-m.paypal.com"
        else:
            self.base_url = "https://api-m.sandbox.paypal.com"

        self.connect_timeout = settings.PAYPAL_CONNECT_TIMEOUT
        self.read_timeout = settings.PAYPAL_READ_TIMEOUT
        self.max_retries = settings.PAYPAL_RETRY_LIMIT

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()
        self._session = requests.Session()

    def format_money_value(self, amount: Money) -> str:
        """Formats Money minor units into decimal string compliant with PayPal requirements."""
        scale = CURRENCY_MINOR_UNITS.get(amount.currency, 2)
        multiplier = Decimal(10 ** scale)
        major_decimal = Decimal(amount.minor_units) / multiplier
        return f"{major_decimal:.{scale}f}"

    def _get_access_token(self) -> str:
        """Acquires and caches an OAuth 2.0 Bearer access token using client credentials."""
        now = time.time()
        if self._token and now < self._token_expires_at:
            return self._token

        with self._token_lock:
            # Re-check under lock
            if self._token and time.time() < self._token_expires_at:
                return self._token

            if not self.client_id or not self.client_secret:
                raise PaymentAuthenticationException(
                    "PayPal credentials (PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET) are missing or unconfigured."
                )

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
                logger.error("Failed to connect to PayPal OAuth endpoint", exc_info=True)
                raise PaymentAuthenticationException(f"Network error during PayPal OAuth: {str(exc)}")

            if response.status_code != 200:
                logger.error("PayPal OAuth failed", extra={"status_code": response.status_code, "body": response.text})
                raise PaymentAuthenticationException(f"PayPal authentication failed with status {response.status_code}")

            try:
                token_data = response.json()
            except Exception:
                raise PaymentAuthenticationException("Invalid JSON returned by PayPal OAuth endpoint")

            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)

            if not access_token:
                raise PaymentAuthenticationException("PayPal response did not contain an access_token.")

            self._token = access_token
            # Cache token with configured safety buffer (default 300s)
            buffer = settings.OAUTH_TOKEN_CACHE_TTL_SECONDS
            self._token_expires_at = time.time() + max(60, expires_in - buffer)
            return self._token

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        idempotent: bool = True
    ) -> Dict[str, Any]:
        """Executes authenticated HTTP request against PayPal API with retry handling on transient errors."""
        token = self._get_access_token()
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)

        attempts = 0
        last_error = None

        while attempts < (self.max_retries if idempotent else 1):
            attempts += 1
            try:
                response = self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                # Check status
                if response.status_code in (200, 201, 204):
                    if response.status_code == 204 or not response.content:
                        return {"status": "SUCCESS"}
                    return response.json()

                # Handle specific HTTP error status codes
                status_code = response.status_code
                error_body = response.text
                try:
                    error_json = response.json()
                except Exception:
                    error_json = {"error": error_body}

                if status_code == 404:
                    raise PaymentOrderNotFoundException(f"Resource at {path} not found")
                elif status_code in (401, 403):
                    # Invalidate cached token on auth failure
                    self._token = None
                    self._token_expires_at = 0.0
                    raise PaymentAuthenticationException(f"PayPal authorization rejected: {error_json.get('message', error_body)}")
                elif status_code == 422:
                    error_name = error_json.get("name", "UNPROCESSABLE_ENTITY")
                    message = error_json.get("message", "Validation error occurred at PayPal")
                    details = error_json.get("details", [])
                    raise PaymentProviderException(f"{error_name}: {message}", status_code=status_code, details={"details": details})
                elif status_code >= 500 and idempotent and attempts < self.max_retries:
                    logger.warning(f"Transient PayPal error {status_code}, retrying attempt {attempts}...", extra={"path": path})
                    time.sleep(0.5 * (2 ** (attempts - 1)))
                    continue
                else:
                    raise PaymentProviderException(
                        f"PayPal API error ({status_code}): {error_json.get('message', error_body)}",
                        status_code=status_code,
                        details=error_json
                    )

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if idempotent and attempts < self.max_retries:
                    logger.warning(f"Network timeout contacting PayPal, retrying attempt {attempts}...", extra={"path": path})
                    time.sleep(0.5 * (2 ** (attempts - 1)))
                    continue
                raise PaymentProviderException(f"Network error contacting PayPal: {str(exc)}")

        if last_error:
            raise PaymentProviderException(f"Max retries exceeded contacting PayPal: {str(last_error)}")
        raise PaymentProviderException("Failed to complete PayPal request")

    # -------------------------------------------------------------------------
    # Orders API v2
    # -------------------------------------------------------------------------

    def create_order(
        self,
        amount: Money,
        reference_id: str,
        return_url: str,
        cancel_url: str,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a PayPal order with intent CAPTURE."""
        if amount.minor_units <= 0:
            raise InvalidPaymentAmountException("Payment amount must be greater than zero")

        formatted_value = self.format_money_value(amount)
        payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": reference_id,
                    "description": f"Donation - {reference_id}",
                    "amount": {
                        "currency_code": amount.currency,
                        "value": formatted_value
                    }
                }
            ],
            "application_context": {
                "brand_name": "CharityChain",
                "landing_page": "BILLING",
                "user_action": "PAY_NOW",
                "return_url": return_url,
                "cancel_url": cancel_url
            }
        }
        if custom_id:
            payload["purchase_units"][0]["custom_id"] = custom_id

        response = self._request("POST", "/v2/checkout/orders", json_data=payload, idempotent=False)
        order_id = response.get("id")

        # Extract approval URL
        approval_url = None
        for link in response.get("links", []):
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break

        return {
            "id": order_id,
            "status": response.get("status", "CREATED"),
            "approval_url": approval_url,
            "raw_response": response
        }

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetches status and details of a PayPal order."""
        return self._request("GET", f"/v2/checkout/orders/{order_id}")

    def capture_order(
        self,
        order_id: str,
        amount: Optional[Money] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Captures payment for an approved order."""
        headers = {}
        if idempotency_key:
            headers["PayPal-Request-Id"] = idempotency_key

        payload = {}
        if amount:
            headers["Content-Type"] = "application/json"
            payload = {
                "amount": {
                    "currency_code": amount.currency,
                    "value": self.format_money_value(amount)
                }
            }

        try:
            response = self._request(
                "POST",
                f"/v2/checkout/orders/{order_id}/capture",
                json_data=payload,
                extra_headers=headers,
                idempotent=False
            )
        except PaymentProviderException as exc:
            # Check for already captured condition (ORDER_ALREADY_CAPTURED)
            details = getattr(exc, "details", {})
            if "ORDER_ALREADY_CAPTURED" in str(details) or exc.status_code == 422:
                # Fetch order to determine capture state
                order = self.get_order(order_id)
                if order.get("status") == "COMPLETED":
                    pu = order.get("purchase_units", [{}])[0]
                    payments = pu.get("payments", {})
                    captures = payments.get("captures", [])
                    if captures:
                        cap = captures[0]
                        return {
                            "id": cap.get("id"),
                            "status": "COMPLETED",
                            "order_id": order_id,
                            "amount": cap.get("amount"),
                            "raw_response": order
                        }
            raise PaymentCaptureFailedException(f"Failed to capture PayPal order: {str(exc)}")

        # Extract capture details
        purchase_units = response.get("purchase_units", [])
        capture_id = None
        capture_status = "UNKNOWN"
        capture_amount = None

        if purchase_units:
            payments = purchase_units[0].get("payments", {})
            captures = payments.get("captures", [])
            if captures:
                first_capture = captures[0]
                capture_id = first_capture.get("id")
                capture_status = first_capture.get("status", "COMPLETED")
                capture_amount = first_capture.get("amount")

        return {
            "id": capture_id or response.get("id"),
            "status": capture_status,
            "order_id": order_id,
            "amount": capture_amount,
            "raw_response": response
        }

    # -------------------------------------------------------------------------
    # Payments API v2 - Captures & Refunds
    # -------------------------------------------------------------------------

    def get_capture(self, capture_id: str) -> Dict[str, Any]:
        """Retrieves details of a settled payment capture."""
        return self._request("GET", f"/v2/payments/captures/{capture_id}")

    def authorize_payment(self, order_id: str) -> Dict[str, Any]:
        """Direct orders use intent CAPTURE; authorizations not used for standard donations."""
        raise NotImplementedError("CharityChain Phase 4 uses immediate capture intent; authorizations not supported.")

    def capture_authorization(self, authorization_id: str, amount: Optional[Money] = None) -> Dict[str, Any]:
        raise NotImplementedError("Direct capture in use.")

    def void_authorization(self, authorization_id: str) -> bool:
        raise NotImplementedError("Direct capture in use.")

    def refund_capture(
        self,
        capture_id: str,
        amount: Money,
        reason: str,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes a partial or full refund on a completed capture."""
        if amount.minor_units <= 0:
            raise InvalidPaymentAmountException("Refund amount must be strictly greater than zero")

        headers = {}
        if idempotency_key:
            headers["PayPal-Request-Id"] = idempotency_key

        payload = {
            "amount": {
                "value": self.format_money_value(amount),
                "currency_code": amount.currency
            },
            "note_to_payer": reason[:255] if reason else "CharityChain Donation Refund"
        }

        try:
            response = self._request(
                "POST",
                f"/v2/payments/captures/{capture_id}/refund",
                json_data=payload,
                extra_headers=headers,
                idempotent=False
            )
        except Exception as exc:
            raise PaymentRefundFailedException(f"PayPal refund failed: {str(exc)}")

        return {
            "id": response.get("id"),
            "status": response.get("status", "COMPLETED"),
            "capture_id": capture_id,
            "amount": response.get("amount"),
            "raw_response": response
        }

    def get_refund(self, refund_id: str) -> Dict[str, Any]:
        """Retrieves details of a processed refund."""
        return self._request("GET", f"/v2/payments/refunds/{refund_id}")

    # -------------------------------------------------------------------------
    # Webhooks API v1 - Cryptographic Signature Verification
    # -------------------------------------------------------------------------

    def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        raw_body: bytes,
        webhook_id: Optional[str] = None
    ) -> bool:
        """
        Cryptographically validates incoming PayPal webhook payloads via PayPal API.
        """
        # Normalize header keys to case-insensitive lookup
        h = {k.upper(): v for k, v in headers.items()}

        auth_algo = h.get("PAYPAL-AUTH-ALGO")
        cert_url = h.get("PAYPAL-CERT-URL")
        transmission_id = h.get("PAYPAL-TRANSMISSION-ID")
        transmission_sig = h.get("PAYPAL-TRANSMISSION-SIG")
        transmission_time = h.get("PAYPAL-TRANSMISSION-TIME")
        target_webhook_id = webhook_id or self.webhook_id

        if not all([auth_algo, cert_url, transmission_id, transmission_sig, transmission_time, target_webhook_id]):
            logger.warning("Missing required PayPal webhook verification headers or webhook_id")
            return False

        try:
            webhook_event = json.loads(raw_body.decode("utf-8"))
        except Exception:
            logger.error("Failed to parse raw webhook body as JSON for verification")
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
            verification_status = res.get("verification_status")
            return verification_status == "SUCCESS"
        except Exception as exc:
            logger.error(f"Error calling PayPal webhook signature verification: {str(exc)}")
            return False
