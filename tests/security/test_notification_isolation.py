"""
test_notification_isolation.py - Security Tests for Multi-Tenant IDOR Isolation
================================================================================
Verifies that notifications, user preferences, and webhook subscriptions enforce
strict multi-tenant and IDOR authorization boundaries.
"""

import uuid
import pytest
from backend.app.core.constants import NotificationCategory, PlatformRole, AccountStatus
from backend.app.core.exceptions import ForbiddenException, NotFoundException
from backend.app.core.security import hash_password
from backend.app.db.models.users import User
from backend.app.services.notification_service import NotificationService
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService


@pytest.fixture
def user_alice(db_session):
    u = User(
        id=str(uuid.uuid4()),
        email=f"alice_{uuid.uuid4().hex[:8]}@example.com",
        full_name="Alice Smith",
        hashed_password=hash_password("Password123!"),
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(u)
    db_session.commit()
    return u


@pytest.fixture
def user_bob(db_session):
    u = User(
        id=str(uuid.uuid4()),
        email=f"bob_{uuid.uuid4().hex[:8]}@example.com",
        full_name="Bob Jones",
        hashed_password=hash_password("Password123!"),
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(u)
    db_session.commit()
    return u


def test_notification_idor_defense(db_session, user_alice, user_bob):
    """Verifies that User Bob cannot view or modify Alice's notification."""
    notif = NotificationService.create_in_app_notification(
        db=db_session,
        user_id=user_alice.id,
        category=NotificationCategory.DONATIONS,
        event_type="donation.captured",
        title="Alice Private Receipt",
        message="Confidential financial message."
    )
    db_session.commit()

    # Bob attempts to read Alice's notification directly -> ForbiddenException
    with pytest.raises(ForbiddenException):
        NotificationService.get_notification(db_session, user_id=user_bob.id, notification_id=notif.id)

    # Bob attempts to mark Alice's notification as read -> ForbiddenException
    with pytest.raises(ForbiddenException):
        NotificationService.mark_as_read(db_session, user_id=user_bob.id, notification_id=notif.id)

    # Bob attempts to archive Alice's notification -> ForbiddenException
    with pytest.raises(ForbiddenException):
        NotificationService.archive_notification(db_session, user_id=user_bob.id, notification_id=notif.id)


def test_webhook_cross_tenant_isolation(db_session, user_alice, user_bob):
    """Verifies that User Bob cannot manage or delete Alice's webhook subscription."""
    sub, _ = WebhookSubscriptionService.create_subscription(
        db=db_session,
        created_by_user_id=user_alice.id,
        endpoint_url="https://alice-ngo.org/hooks"
    )
    db_session.commit()

    # Bob attempts to rotate Alice's webhook secret
    with pytest.raises(ForbiddenException):
        WebhookSubscriptionService.rotate_secret(
            db=db_session,
            subscription_id=sub.id,
            requesting_user_id=user_bob.id
        )

    # Bob attempts to delete Alice's webhook
    with pytest.raises(ForbiddenException):
        WebhookSubscriptionService.delete_subscription(
            db=db_session,
            subscription_id=sub.id,
            requesting_user_id=user_bob.id
        )

