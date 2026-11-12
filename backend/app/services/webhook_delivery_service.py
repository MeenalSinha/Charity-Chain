"""
webhook_delivery_service.py - Outbound Webhook Delivery & HMAC Signing Engine
=============================================================================
Manages secure HTTP dispatch of external webhooks:
- HMAC-SHA256 signature generation and timestamp-based replay protection
- SSRF destination re-validation on every request (anti-DNS rebinding)
- HTTP timeout enforcement and response size bounds
- Exponential backoff with jitter, retry attempts, and dead-letter queue
- Automatic subscription suspension upon repeated failure thresholds
"""

import hmac
import time
import json
import uuid
import random
import hashlib
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.config import settings
from backend.app.core.constants import WebhookDeliveryStatus
from backend.app.core.events import DomainEvent
from backend.app.core.canonical_json import canonicalize_json
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import get_logger
from backend.app.core.ssrf_protection import validate_webhook_url
from backend.app.db.models.notifications import (
    WebhookSubscription, WebhookDelivery, WebhookDeliveryAttempt
)
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService

logger = get_logger("services.webhooks.delivery")


class WebhookDeliveryService:
    """Service orchestrating cryptographic signing, HTTP dispatch, and resilient retries for webhooks."""

    @staticmethod
    def compute_hmac_signature(secret: str, timestamp: int, payload_bytes: bytes) -> str:
        """
        Computes HMAC-SHA256 signature over {timestamp}.{payload_bytes}.
        Returns raw hex digest.
        """
        signed_payload = f"{timestamp}.".encode("utf-8") + payload_bytes
        return hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()

    @classmethod
    def compute_signature(cls, payload: str | bytes, secret: str, timestamp: Optional[int] = None) -> str:
        """Convenience method returning full 't={timestamp},v1={sig}' header value."""
        ts = timestamp if timestamp is not None else int(time.time())
        p_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        sig_hex = cls.compute_hmac_signature(secret, ts, p_bytes)
        return f"t={ts},v1={sig_hex}"

    @classmethod
    def verify_hmac_signature(
        cls,
        secret: str,
        signature_header: str,
        payload_bytes: bytes,
        tolerance_seconds: int = 300
    ) -> bool:
        """
        Validates an incoming CharityChain-Signature header: 't={timestamp},v1={hex_sig}'.
        Includes constant-time comparison and timestamp replay defense.
        """
        if not signature_header or not secret:
            return False

        try:
            parts = dict(item.split("=", 1) for item in signature_header.split(","))
            timestamp = int(parts.get("t", "0"))
            v1_sig = parts.get("v1", "")

            # Replay defense: verify timestamp is within tolerance window
            current_time = int(time.time())
            if abs(current_time - timestamp) > tolerance_seconds:
                logger.warning(f"Webhook signature rejected: timestamp {timestamp} outside tolerance ({tolerance_seconds}s)")
                return False

            expected_sig = cls.compute_hmac_signature(secret, timestamp, payload_bytes)
            return hmac.compare_digest(expected_sig, v1_sig)
        except Exception as exc:
            logger.error(f"Error validating webhook signature: {str(exc)}")
            return False

    @classmethod
    def verify_signature(cls, payload: str | bytes, signature_header: str, secret: str, tolerance_seconds: int = 300) -> bool:
        """Convenience signature verification method accepting str or bytes."""
        p_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        return cls.verify_hmac_signature(secret, signature_header, p_bytes, tolerance_seconds)

    @staticmethod
    def enqueue_deliveries_for_event(db: Session, event: DomainEvent) -> List[WebhookDelivery]:
        """
        Evaluates active webhook subscriptions and enqueues WebhookDelivery records.
        """
        # Find all active subscriptions
        query = select(WebhookSubscription).where(WebhookSubscription.is_active.is_(True))
        subscriptions = list(db.execute(query).scalars().all())

        deliveries = []
        for sub in subscriptions:
            # Check tenant isolation: subscription must be platform-wide or match event org
            if sub.organization_id and event.organization_id and sub.organization_id != event.organization_id:
                continue

            # Check event type matching
            events = sub.subscribed_events or []
            is_matched = False
            for pattern in events:
                if pattern == "*" or pattern == event.event_type:
                    is_matched = True
                    break
                if pattern.endswith(".*") and event.event_type.startswith(pattern[:-2]):
                    is_matched = True
                    break

            if not is_matched:
                continue

            delivery = WebhookDelivery(
                id=str(uuid.uuid4()),
                subscription_id=sub.id,
                event_id=event.event_id,
                event_type=event.event_type,
                payload_json=event.to_json(),
                status=WebhookDeliveryStatus.PENDING,
                retry_count=0,
                max_retries=getattr(settings, "WEBHOOK_MAX_RETRIES", 5),
                next_retry_at=datetime.now(timezone.utc)
            )
            db.add(delivery)
            deliveries.append(delivery)

        if deliveries:
            db.flush()
            logger.info(f"Enqueued {len(deliveries)} webhook deliveries for event {event.event_id} ({event.event_type})")

        return deliveries

    @classmethod
    def deliver_webhook(cls, db: Session, delivery_id: str) -> bool:
        """
        Executes outbound HTTP dispatch of a webhook delivery:
        - Validates SSRF safety
        - Computes HMAC-SHA256 signature
        - Dispatches POST request with timeout and no auto-redirects
        - Tracks attempt telemetry and manages retry schedules
        """
        delivery = db.execute(
            select(WebhookDelivery).where(WebhookDelivery.id == delivery_id)
        ).scalar_one_or_none()

        if not delivery:
            logger.error(f"Webhook delivery {delivery_id} not found.")
            return False

        sub = delivery.subscription
        if not sub or not sub.is_active:
            logger.info(f"Subscription {delivery.subscription_id} inactive or suspended. Skipping delivery.")
            delivery.status = WebhookDeliveryStatus.FAILED
            db.commit()
            return False

        # Get active signing secret
        secret = WebhookSubscriptionService.get_plaintext_secret(db, sub.id)
        if not secret:
            logger.error(f"No active secret found for subscription {sub.id}. Cannot sign webhook.")
            delivery.status = WebhookDeliveryStatus.FAILED
            db.commit()
            return False

        # 1. SSRF Safety Re-validation before dispatch
        try:
            validate_webhook_url(sub.endpoint_url)
        except Exception as exc:
            logger.error(f"SSRF validation blocked outbound dispatch to {sub.endpoint_url}: {str(exc)}")
            cls._record_attempt(
                db=db,
                delivery=delivery,
                sub=sub,
                status_code=None,
                response_headers=None,
                response_body=None,
                error_message=f"SSRF Blocked: {str(exc)}",
                duration_ms=0,
                is_success=False
            )
            return False

        # 2. Prepare payload & HMAC signature
        payload_bytes = delivery.payload_json.encode("utf-8")
        timestamp = int(time.time())
        signature_hex = cls.compute_hmac_signature(secret, timestamp, payload_bytes)
        signature_header = f"t={timestamp},v1={signature_hex}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CharityChain-Webhook-Dispatcher/1.0",
            "CharityChain-Signature": signature_header,
            "CharityChain-Event-Id": delivery.event_id,
            "CharityChain-Event-Type": delivery.event_type
        }

        # 3. HTTP Dispatch
        start_time = time.time()
        timeout = getattr(settings, "WEBHOOK_TIMEOUT_SECONDS", 10.0)

        try:
            response = requests.post(
                sub.endpoint_url,
                data=payload_bytes,
                headers=headers,
                timeout=timeout,
                allow_redirects=False  # Crucial SSRF defense: never follow redirects automatically
            )
            duration_ms = int((time.time() - start_time) * 1000)
            is_success = 200 <= response.status_code < 300
            truncated_body = response.text[:1024] if response.text else None
            headers_json = json.dumps(dict(response.headers))

            cls._record_attempt(
                db=db,
                delivery=delivery,
                sub=sub,
                status_code=response.status_code,
                response_headers=headers_json,
                response_body=truncated_body,
                error_message=None if is_success else f"HTTP Status {response.status_code}",
                duration_ms=duration_ms,
                is_success=is_success
            )
            return is_success

        except requests.Timeout as exc:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Webhook dispatch timeout to {sub.endpoint_url} after {duration_ms}ms")
            cls._record_attempt(
                db=db,
                delivery=delivery,
                sub=sub,
                status_code=None,
                response_headers=None,
                response_body=None,
                error_message=f"Request Timeout: {str(exc)}",
                duration_ms=duration_ms,
                is_success=False
            )
            return False

        except Exception as exc:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Webhook dispatch network error to {sub.endpoint_url}: {str(exc)}")
            cls._record_attempt(
                db=db,
                delivery=delivery,
                sub=sub,
                status_code=None,
                response_headers=None,
                response_body=None,
                error_message=f"Network Error: {str(exc)}",
                duration_ms=duration_ms,
                is_success=False
            )
            return False

    @staticmethod
    def _record_attempt(
        db: Session,
        delivery: WebhookDelivery,
        sub: WebhookSubscription,
        status_code: Optional[int],
        response_headers: Optional[str],
        response_body: Optional[str],
        error_message: Optional[str],
        duration_ms: int,
        is_success: bool
    ):
        """Records attempt telemetry, updates delivery state, and manages retries."""
        now = datetime.now(timezone.utc)
        delivery.retry_count += 1

        attempt = WebhookDeliveryAttempt(
            id=str(uuid.uuid4()),
            delivery_id=delivery.id,
            attempt_number=delivery.retry_count,
            status_code=status_code,
            response_headers_json=response_headers,
            response_body=response_body,
            error_message=error_message,
            execution_duration_ms=duration_ms,
            attempted_at=now
        )
        db.add(attempt)

        if is_success:
            delivery.status = WebhookDeliveryStatus.DELIVERED
            delivery.completed_at = now
            delivery.next_retry_at = None
            sub.consecutive_failures = 0
            logger.info(f"Webhook delivery {delivery.id} succeeded (status={status_code}) in {duration_ms}ms")
        else:
            sub.consecutive_failures += 1
            # Check auto-suspend threshold
            suspend_threshold = getattr(settings, "WEBHOOK_AUTO_SUSPEND_THRESHOLD", 10)
            if sub.consecutive_failures >= suspend_threshold:
                sub.is_active = False
                sub.suspended_at = now
                logger.error(
                    f"Webhook subscription {sub.id} automatically SUSPENDED after {sub.consecutive_failures} consecutive failures."
                )

            # Check retry budget
            if delivery.retry_count >= delivery.max_retries:
                delivery.status = WebhookDeliveryStatus.DEAD_LETTER
                delivery.next_retry_at = None
                logger.error(f"Webhook delivery {delivery.id} moved to DEAD_LETTER after {delivery.retry_count} retries.")
            else:
                delivery.status = WebhookDeliveryStatus.FAILED
                base_sec = getattr(settings, "WEBHOOK_BACKOFF_BASE_SECONDS", 10.0)
                max_sec = getattr(settings, "WEBHOOK_BACKOFF_MAX_SECONDS", 86400.0)
                exponential = min(max_sec, base_sec * (2 ** (delivery.retry_count - 1)))
                jitter = random.uniform(0, min(10.0, exponential * 0.2))
                wait_seconds = exponential + jitter
                delivery.next_retry_at = now + timedelta(seconds=wait_seconds)
                logger.warning(
                    f"Webhook delivery {delivery.id} failed attempt {delivery.retry_count}/{delivery.max_retries}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )

        db.commit()

    @classmethod
    def manual_retry(cls, db: Session, delivery_id: str, requesting_user_id: str) -> bool:
        """Manually re-triggers a delivery attempt for a failed or dead-lettered webhook."""
        delivery = db.execute(
            select(WebhookDelivery).where(WebhookDelivery.id == delivery_id)
        ).scalar_one_or_none()

        if not delivery:
            raise NotFoundException(f"Webhook delivery {delivery_id} not found.")

        delivery.status = WebhookDeliveryStatus.PENDING
        delivery.next_retry_at = datetime.now(timezone.utc)
        db.commit()

        return cls.deliver_webhook(db, delivery_id)
