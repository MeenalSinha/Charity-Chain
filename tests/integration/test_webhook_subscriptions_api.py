"""
test_webhook_subscriptions_api.py - Integration Tests for Webhook Subscriptions
================================================================================
Verifies webhook registration, secret generation/encryption, secret rotation,
and subscription deletion.
"""

import uuid
import pytest
from backend.app.core.constants import WebhookSubscriptionStatus
from backend.app.db.models.notifications import WebhookSubscription, WebhookEndpointSecret
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService


def test_webhook_subscription_lifecycle_and_secret_encryption(db_session):
    """Verifies that webhook secrets are encrypted in DB and only returned plaintext on creation."""
    user_id = str(uuid.uuid4())
    endpoint = "https://partner-ngo.org/events"

    sub, plaintext_secret = WebhookSubscriptionService.create_subscription(
        db=db_session,
        created_by_user_id=user_id,
        endpoint_url=endpoint,
        description="Primary Integration Webhook",
        subscribed_events=["donation.captured", "certificate.issued"]
    )
    db_session.commit()

    assert sub.id is not None
    assert sub.status == WebhookSubscriptionStatus.ACTIVE
    assert sub.endpoint_url == endpoint
    assert plaintext_secret.startswith("whsec_")

    # In database, secret must NOT be stored in plaintext
    secret_record = db_session.query(WebhookEndpointSecret).filter(
        WebhookEndpointSecret.subscription_id == sub.id
    ).first()
    assert secret_record is not None
    assert secret_record.encrypted_secret != plaintext_secret

    # Verify get_plaintext_secret matches the plaintext secret
    decrypted = WebhookSubscriptionService.get_plaintext_secret(db_session, sub.id)
    assert decrypted == plaintext_secret


def test_webhook_secret_rotation(db_session):
    """Verifies that rotating a secret invalidates the old secret and activates a new one."""
    user_id = str(uuid.uuid4())
    sub, secret_1 = WebhookSubscriptionService.create_subscription(
        db=db_session,
        created_by_user_id=user_id,
        endpoint_url="https://partner-ngo.org/events2"
    )
    db_session.commit()

    # Rotate secret
    secret_2 = WebhookSubscriptionService.rotate_secret(db_session, sub.id, requesting_user_id=user_id)
    db_session.commit()

    assert secret_2.startswith("whsec_")
    assert secret_1 != secret_2

    # Active secret must now be secret_2
    current_active = WebhookSubscriptionService.get_plaintext_secret(db_session, sub.id)
    assert current_active == secret_2


def test_webhook_subscription_deletion(db_session):
    """Verifies deletion removes subscription and associated encrypted secrets."""
    user_id = str(uuid.uuid4())
    sub, _ = WebhookSubscriptionService.create_subscription(
        db=db_session,
        created_by_user_id=user_id,
        endpoint_url="https://partner-ngo.org/events3"
    )
    db_session.commit()

    sub_id = sub.id
    WebhookSubscriptionService.delete_subscription(db_session, sub_id, requesting_user_id=user_id)
    db_session.commit()

    # Subscription and secrets should be gone
    assert db_session.query(WebhookSubscription).filter(WebhookSubscription.id == sub_id).first() is None
    assert db_session.query(WebhookEndpointSecret).filter(WebhookEndpointSecret.subscription_id == sub_id).count() == 0
