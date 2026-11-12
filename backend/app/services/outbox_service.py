"""
outbox_service.py - Transactional Outbox Persistence & Retry Service
===================================================================
Manages database-backed outbox records, guaranteeing atomic event creation
alongside domain changes and supporting resilient retry cycles.
"""

import uuid
import random
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_

from backend.app.core.config import settings
from backend.app.core.constants import OutboxStatus
from backend.app.core.events import DomainEvent
from backend.app.core.logging import get_logger
from backend.app.db.models.notifications import NotificationOutbox

logger = get_logger("services.outbox")


def compute_backoff_seconds(retry_count: int, base_seconds: float = 10.0, max_seconds: float = 3600.0) -> float:
    """Calculates exponential backoff delay with 0-20% jitter."""
    exponential = min(max_seconds, base_seconds * (2 ** max(0, retry_count - 1)))
    jitter = random.uniform(0, min(10.0, exponential * 0.2))
    return exponential + jitter


class OutboxService:
    """Service managing transactional event outbox persistence and dispatch tracking."""

    @staticmethod
    def record_event(db: Session, event: DomainEvent) -> NotificationOutbox:
        """
        Persists a domain event to the notification outbox within the caller's transaction.
        Idempotent: if an outbox record with the same idempotency_key exists, returns it.
        """
        existing = db.execute(
            select(NotificationOutbox).where(NotificationOutbox.idempotency_key == event.idempotency_key)
        ).scalar_one_or_none()

        if existing:
            logger.info(f"Outbox event already recorded with idempotency key: {event.idempotency_key}")
            return existing

        outbox_record = NotificationOutbox(
            id=str(uuid.uuid4()),
            event_id=event.event_id,
            event_type=event.event_type,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            organization_id=event.organization_id,
            actor_user_id=event.actor_user_id,
            correlation_id=event.correlation_id,
            idempotency_key=event.idempotency_key,
            payload_json=event.to_json(),
            status=OutboxStatus.PENDING,
            retry_count=0,
            max_retries=getattr(settings, "OUTBOX_MAX_RETRIES", 5),
            next_retry_at=datetime.now(timezone.utc)
        )
        db.add(outbox_record)
        db.flush()
        return outbox_record

    @staticmethod
    def get_pending_events(db: Session, limit: int = 50) -> List[NotificationOutbox]:
        """
        Selects pending or retryable outbox events eligible for dispatch.
        """
        now = datetime.now(timezone.utc)
        query = select(NotificationOutbox).where(
            or_(
                NotificationOutbox.status == OutboxStatus.PENDING,
                and_(
                    NotificationOutbox.status == OutboxStatus.FAILED,
                    NotificationOutbox.next_retry_at <= now,
                    NotificationOutbox.retry_count < NotificationOutbox.max_retries
                )
            )
        ).order_by(NotificationOutbox.created_at.asc()).limit(limit)

        return list(db.execute(query).scalars().all())

    @staticmethod
    def mark_processing(db: Session, outbox_id: str) -> Optional[NotificationOutbox]:
        """Marks an outbox record as currently PROCESSING."""
        record = db.execute(
            select(NotificationOutbox).where(NotificationOutbox.id == outbox_id)
        ).scalar_one_or_none()

        if record:
            record.status = OutboxStatus.PROCESSING
            record.updated_at = datetime.now(timezone.utc)
            db.flush()
        return record

    @staticmethod
    def mark_processed(db: Session, outbox_id: str) -> Optional[NotificationOutbox]:
        """Marks an outbox record as successfully PROCESSED."""
        record = db.execute(
            select(NotificationOutbox).where(NotificationOutbox.id == outbox_id)
        ).scalar_one_or_none()

        if record:
            now = datetime.now(timezone.utc)
            record.status = OutboxStatus.PROCESSED
            record.processed_at = now
            record.updated_at = now
            record.error_message = None
            db.flush()
        return record

    @staticmethod
    def mark_failed(
        db: Session,
        outbox_id: str,
        error_message: str,
        base_seconds: float = 10.0,
        max_seconds: float = 3600.0,
        max_retry_count: Optional[int] = None
    ) -> Optional[NotificationOutbox]:
        """
        Increments retry count, calculates exponential backoff with jitter, and sets next retry.
        If retry count reaches max_retries, transitions to DEAD_LETTER.
        """
        record = db.execute(
            select(NotificationOutbox).where(NotificationOutbox.id == outbox_id)
        ).scalar_one_or_none()

        if not record:
            return None

        effective_limit = max_retry_count if max_retry_count is not None else record.max_retries
        now = datetime.now(timezone.utc)
        record.retry_count += 1
        record.error_message = error_message[:1000]
        record.updated_at = now

        if record.retry_count >= effective_limit:
            record.status = OutboxStatus.DEAD_LETTER
            record.next_retry_at = None
            logger.error(f"Outbox event {record.event_id} moved to DEAD_LETTER after {record.retry_count} retries.")
        else:
            record.status = OutboxStatus.FAILED
            wait_seconds = compute_backoff_seconds(record.retry_count, base_seconds, max_seconds)
            record.next_retry_at = now + timedelta(seconds=wait_seconds)
            logger.warning(
                f"Outbox event {record.event_id} failed attempt {record.retry_count}/{effective_limit}. "
                f"Next retry in {wait_seconds:.1f}s."
            )

        db.flush()
        return record

    @staticmethod
    def mark_dead_letter(db: Session, outbox_id: str, error_message: str) -> Optional[NotificationOutbox]:
        """Explicitly transitions an event to DEAD_LETTER (unrecoverable error)."""
        record = db.execute(
            select(NotificationOutbox).where(NotificationOutbox.id == outbox_id)
        ).scalar_one_or_none()

        if record:
            now = datetime.now(timezone.utc)
            record.status = OutboxStatus.DEAD_LETTER
            record.error_message = error_message[:1000]
            record.updated_at = now
            record.next_retry_at = None
            db.flush()
        return record
