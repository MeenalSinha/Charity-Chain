"""
test_notification_worker.py - Integration Tests for Transactional Outbox Worker
==============================================================================
Verifies the complete outbox dispatch loop: event recording, background worker
polling, in-app notification creation, email dispatch, and outbox completion.
"""

import uuid
import pytest
from backend.app.core.constants import OutboxStatus, PlatformRole, AccountStatus, DomainEventType
from backend.app.core.events import DomainEvent
from backend.app.core.security import hash_password
from backend.app.db.models.users import User
from backend.app.db.models.notifications import Notification, NotificationOutbox
from backend.app.integrations.notifications.dev_sink import DevNotificationSink
from backend.app.integrations.notifications.factory import set_email_provider
from backend.app.services.outbox_service import OutboxService
from backend.app.workers.notification_worker import NotificationWorker


@pytest.fixture
def test_donor(db_session):
    user = User(
        id=str(uuid.uuid4()),
        email=f"donor_{uuid.uuid4().hex[:8]}@example.com",
        full_name="Grace Hopper",
        hashed_password=hash_password("Password123!"),
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_outbox_worker_processes_donation_event(db_session, test_donor):
    """Verifies that an outbox event is processed into an in-app notification and email."""
    # 1. Configure DevNotificationSink
    sink = DevNotificationSink()
    set_email_provider(sink)

    # 2. Record donation.captured event to Outbox
    event = DomainEvent.create(
        event_type=DomainEventType.PAYMENT_CAPTURED.value,
        entity_type="payment_capture",
        entity_id="cap-001",
        actor_user_id=test_donor.id,
        payload={
            "donation_id": "don-001",
            "donor_id": test_donor.id,
            "donor_email": test_donor.email,
            "donor_name": test_donor.full_name,
            "project_title": "Clean Water Well",
            "amount_minor_units": 7500,
            "currency": "USD"
        }
    )
    outbox_entry = OutboxService.record_event(db_session, event)
    db_session.commit()
    assert outbox_entry.status == OutboxStatus.PENDING

    # 3. Execute Worker batch cycle
    # Define session factory returning the current active test db_session
    class MockSessionFactory:
        def __call__(self):
            return db_session

    worker = NotificationWorker(session_factory=MockSessionFactory(), poll_interval_seconds=0.1, batch_size=10)
    stats = worker.process_outbox_batch(db=db_session)

    assert stats["processed"] >= 1
    assert stats["delivered"] >= 1

    # 4. Verify Outbox Record is now PROCESSED
    db_session.refresh(outbox_entry)
    assert outbox_entry.status == OutboxStatus.PROCESSED
    assert outbox_entry.processed_at is not None

    # 5. Verify In-App Notification was generated
    notif = db_session.query(Notification).filter(Notification.user_id == test_donor.id).first()
    assert notif is not None
    assert "Donation" in notif.title
    assert "Clean Water Well" in notif.message
    assert notif.is_read is False

    # 6. Verify Email was recorded by sink
    assert len(sink.captured_messages) >= 1
    captured = sink.captured_messages[-1]
    assert captured["to_email"] == test_donor.email
    assert "$75.00" in captured["subject"] or "$75.00" in captured["html_body"]
