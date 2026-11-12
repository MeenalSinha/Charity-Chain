"""
test_notification_center_api.py - Integration Tests for In-App Notification Center
==================================================================================
Tests user notification listing, unread counts, status transitions (read/archive),
and user preference management.
"""

import uuid
import pytest
from backend.app.core.constants import NotificationCategory, PlatformRole, AccountStatus
from backend.app.core.security import hash_password
from backend.app.db.models.users import User
from backend.app.services.notification_service import NotificationService


@pytest.fixture
def test_user(db_session):
    user = User(
        id=str(uuid.uuid4()),
        email=f"user_{uuid.uuid4().hex[:8]}@example.org",
        full_name="Notification Test User",
        hashed_password=hash_password("ValidPassword123!"),
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_notification_lifecycle_and_unread_counts(db_session, test_user):
    """Verifies creation, unread counting, mark as read, and archiving."""
    # 1. Unread count starts at 0
    initial_unread = NotificationService.get_unread_count(db_session, test_user.id)
    assert initial_unread == 0

    # 2. Create 2 notifications
    n1 = NotificationService.create_in_app_notification(
        db=db_session,
        user_id=test_user.id,
        category=NotificationCategory.DONATIONS,
        event_type="donation.captured",
        title="Donation Confirmed",
        message="Thank you for your generous gift of $50."
    )
    n2 = NotificationService.create_in_app_notification(
        db=db_session,
        user_id=test_user.id,
        category=NotificationCategory.CERTIFICATES,
        event_type="certificate.issued",
        title="Impact Certificate Issued",
        message="Your digital certificate CC-2026-001 is ready."
    )
    db_session.commit()

    assert NotificationService.get_unread_count(db_session, test_user.id) == 2

    # 3. Mark n1 as read
    NotificationService.mark_as_read(db_session, test_user.id, n1.id)
    assert NotificationService.get_unread_count(db_session, test_user.id) == 1

    # 4. Mark all as read
    NotificationService.mark_all_as_read(db_session, test_user.id)
    assert NotificationService.get_unread_count(db_session, test_user.id) == 0

    # 5. Archive n2
    NotificationService.archive_notification(db_session, test_user.id, n2.id)

    # 6. List active notifications (include_archived=False)
    active, total_active = NotificationService.list_notifications(
        db_session, user_id=test_user.id, include_archived=False
    )
    assert len(active) == 1
    assert active[0].id == n1.id


def test_notification_preferences_safety(db_session, test_user):
    """Verifies that security alerts cannot be disabled even if requested."""
    prefs = NotificationService.get_or_create_preferences(db_session, test_user.id)
    assert prefs.email_enabled is True
    assert prefs.security_alerts is True

    # Attempt to disable security alerts along with other categories
    updated_prefs = NotificationService.update_preferences(
        db_session,
        user_id=test_user.id,
        preferences_data={
            "email_enabled": False,
            "donation_updates": False,
            "security_alerts": False  # Must remain True
        }
    )

    assert updated_prefs.email_enabled is False
    assert updated_prefs.donation_updates is False
    assert updated_prefs.security_alerts is True  # Invariant strictly preserved

