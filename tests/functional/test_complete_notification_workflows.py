"""
test_complete_notification_workflows.py - Functional Tests for Multi-Channel Communications
============================================================================================
Tests end-to-end notification journeys:
1. Donation capture emits outbox event -> In-app notification + Email receipt delivered.
2. Certificate issuance emits outbox event -> In-app notification + Outbound Webhook delivered with HMAC signature.
"""

import time
import uuid
from unittest.mock import patch, MagicMock
import pytest

from backend.app.core.constants import (
    PlatformRole, AccountStatus, OutboxStatus, DomainEventType,
    WebhookDeliveryStatus
)
from backend.app.core.events import DomainEvent
from backend.app.core.security import hash_password
from backend.app.db.models.users import User
from backend.app.db.models.notifications import Notification, NotificationOutbox, WebhookDelivery
from backend.app.integrations.notifications.dev_sink import DevNotificationSink
from backend.app.integrations.notifications.factory import set_email_provider
from backend.app.services.outbox_service import OutboxService
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService
from backend.app.services.webhook_delivery_service import WebhookDeliveryService
from backend.app.workers.notification_worker import NotificationWorker


@pytest.fixture
def donor_user(db_session):
    u = User(
        id=str(uuid.uuid4()),
        email=f"donor_{uuid.uuid4().hex[:8]}@example.org",
        full_name="Evelyn Donor",
        hashed_password=hash_password("Password123!"),
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(u)
    db_session.commit()
    return u


def test_end_to_end_donation_notification_flow(db_session, donor_user):
    """End-to-end test: donation captured -> outbox -> in-app notification + dev-sink email."""
    sink = DevNotificationSink()
    set_email_provider(sink)

    event = DomainEvent.create(
        event_type=DomainEventType.PAYMENT_CAPTURED.value,
        entity_type="payment_capture",
        entity_id="cap-func-001",
        actor_user_id=donor_user.id,
        payload={
            "donation_id": "don-func-001",
            "donor_id": donor_user.id,
            "donor_email": donor_user.email,
            "donor_name": donor_user.full_name,
            "project_title": "Clean Water Wells in Tigray",
            "amount_minor_units": 10000,
            "currency": "USD"
        }
    )
    OutboxService.record_event(db_session, event)
    db_session.commit()

    class MockSessionFactory:
        def __call__(self):
            return db_session

    worker = NotificationWorker(session_factory=MockSessionFactory(), poll_interval_seconds=0.1, batch_size=10)
    worker.process_outbox_batch(db=db_session)

    # 1. In-app notification
    notif = db_session.query(Notification).filter(Notification.user_id == donor_user.id).first()
    assert notif is not None
    assert "Clean Water Wells" in notif.message

    # 2. Email delivered
    assert len(sink.captured_messages) >= 1
    email = sink.captured_messages[-1]
    assert email["to_email"] == donor_user.email
    assert "$100.00" in email["subject"] or "$100.00" in email["html_body"]


def test_end_to_end_certificate_webhook_delivery_flow(db_session, donor_user):
    """End-to-end test: certificate issued -> outbox -> webhook delivery with valid HMAC."""
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]):
        sub, secret = WebhookSubscriptionService.create_subscription(
            db=db_session,
            created_by_user_id=donor_user.id,
            endpoint_url="https://api.partner-ngo.org/charitychain/webhooks",
            description="Certificate listener",
            subscribed_events=[DomainEventType.CERTIFICATE_ISSUED.value]
        )
    db_session.commit()

    event = DomainEvent.create(
        event_type=DomainEventType.CERTIFICATE_ISSUED.value,
        entity_type="impact_certificate",
        entity_id="cert-func-001",
        actor_user_id=donor_user.id,
        payload={
            "certificate_id": "cert-func-001",
            "certificate_number": "CC-2026-CERT-001",
            "verification_id": "v-test-cert",
            "verification_url": "https://charitychain.org/verify/v-test-cert",
            "recipient_name": donor_user.full_name,
            "donor_id": donor_user.id,
            "donor_email": donor_user.email,
            "project_title": "Reforestation Project",
            "impact_summary": "Planted 250 native saplings."
        }
    )
    OutboxService.record_event(db_session, event)
    db_session.commit()

    captured_requests = []

    def mock_requests_post(url, headers, data, timeout, allow_redirects):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        captured_requests.append({
            "url": url,
            "headers": headers,
            "data": data
        })
        return mock_resp

    with patch("requests.post", side_effect=mock_requests_post), patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]):
        class MockSessionFactory:
            def __call__(self):
                return db_session

        worker = NotificationWorker(session_factory=MockSessionFactory(), poll_interval_seconds=0.1, batch_size=10)
        worker.process_outbox_batch(db=db_session)

    # Verify webhook delivery was recorded
    delivery = db_session.query(WebhookDelivery).filter(WebhookDelivery.subscription_id == sub.id).first()
    assert delivery is not None
    assert delivery.status == WebhookDeliveryStatus.DELIVERED
    assert len(delivery.attempts) >= 1
    assert delivery.attempts[0].status_code == 200

    # Verify captured HTTP dispatch and HMAC-SHA256 signature
    assert len(captured_requests) >= 1
    req = captured_requests[0]
    assert req["url"] == sub.endpoint_url
    assert "CharityChain-Signature" in req["headers"]

    sig_header = req["headers"]["CharityChain-Signature"]
    raw_body = req["data"]
    # Verify HMAC verification passes with the subscription's secret
    is_valid = WebhookDeliveryService.verify_signature(raw_body, sig_header, secret)
    assert is_valid is True
