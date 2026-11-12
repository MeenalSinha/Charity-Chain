"""
test_outbox_retry.py - Unit Tests for Outbox Exponential Backoff & Dead-Letter States
=====================================================================================
Verifies retry backoff calculations, jitter boundaries, and failure state transitions.
"""

from datetime import datetime, timezone
import pytest
from backend.app.core.constants import OutboxStatus
from backend.app.services.outbox_service import OutboxService, compute_backoff_seconds


def test_exponential_backoff_calculation():
    """Verifies exponential backoff with jitter stays within expected ranges."""
    # attempt 1: base 5s -> min 5s, max 6s (with jitter <= 20%)
    delay1 = compute_backoff_seconds(1, base_seconds=5, max_seconds=300)
    assert 5.0 <= delay1 <= 6.5

    # attempt 2: base 10s -> min 10s, max 13s
    delay2 = compute_backoff_seconds(2, base_seconds=5, max_seconds=300)
    assert 10.0 <= delay2 <= 13.0

    # attempt 3: base 20s -> min 20s, max 26s
    delay3 = compute_backoff_seconds(3, base_seconds=5, max_seconds=300)
    assert 20.0 <= delay3 <= 26.0

    # high attempt: capped at max_seconds (300)
    delay_capped = compute_backoff_seconds(10, base_seconds=5, max_seconds=300)
    assert delay_capped <= 360.0


def test_outbox_dead_letter_on_max_attempts(db_session):
    """Verifies that exceeding max_retry_count marks outbox entry as DEAD_LETTER."""
    from backend.app.core.events import DomainEvent
    from backend.app.db.models.notifications import NotificationOutbox

    event = DomainEvent.create(
        event_type="test.retry.event",
        entity_type="test",
        entity_id="test-dlq",
        payload={"data": "sample"}
    )
    outbox_entry = OutboxService.record_event(db_session, event)
    db_session.commit()

    # Fail up to max_retry_count - 1 (default max 5)
    for attempt in range(1, 5):
        updated = OutboxService.mark_failed(
            db_session,
            outbox_entry.id,
            error_message=f"Transient failure {attempt}",
            max_retry_count=5
        )
        assert updated.status == OutboxStatus.FAILED
        assert updated.attempt_count == attempt
        assert updated.next_retry_at is not None

    # 5th failure should trigger DEAD_LETTER
    final_entry = OutboxService.mark_failed(
        db_session,
        outbox_entry.id,
        error_message="Final fatal failure",
        max_retry_count=5
    )
    assert final_entry.status == OutboxStatus.DEAD_LETTER
    assert final_entry.attempt_count == 5
    assert "Final fatal failure" in final_entry.last_error
