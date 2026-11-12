"""
notification_worker.py - Asynchronous Event Outbox & Communication Worker
=========================================================================
Polls pending outbox events and orchestrates multi-channel dispatch:
1. In-app notification creation with user preference filtering
2. Transactional email rendering and SMTP/Dev delivery
3. Outbound webhook subscription matching, signing, and dispatch
4. Resilient failure recovery, exponential backoff, and dead-lettering
"""

import uuid
import time
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.constants import NotificationCategory, NotificationChannel, PlatformRole
from backend.app.core.events import DomainEvent
from backend.app.core.logging import get_logger
from backend.app.db.session import SessionLocal
from backend.app.db.models.users import User, OrganizationMembership
from backend.app.db.models.notifications import NotificationOutbox, NotificationDeliveryAttempt
from backend.app.integrations.notifications.factory import get_email_provider
from backend.app.services.outbox_service import OutboxService
from backend.app.services.notification_service import NotificationService
from backend.app.services.email_template_service import EmailTemplateService
from backend.app.services.webhook_delivery_service import WebhookDeliveryService

logger = get_logger("workers.notifications")


class BatchProcessingResult(dict):
    """Dictionary holding batch metrics that also behaves as an integer count of processed events."""
    def __int__(self) -> int:
        return self.get("processed", 0)

    def __bool__(self) -> bool:
        return self.get("processed", 0) > 0


class NotificationWorker:
    """Asynchronous background worker dispatching outbox events to in-app, email, and webhooks."""

    def __init__(
        self,
        db_factory=None,
        session_factory=None,
        poll_interval_seconds: float = 5.0,
        batch_size: int = 50
    ):
        self.db_factory = session_factory or db_factory or SessionLocal
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self._running = False

    def process_outbox_batch(self, db: Optional[Session] = None, limit: Optional[int] = None) -> BatchProcessingResult:
        """
        Executes a single processing cycle over pending outbox events.
        Returns a BatchProcessingResult with processed, delivered, and failed metrics.
        """
        batch_limit = limit or self.batch_size
        if db is not None:
            return self._process_batch_internal(db, batch_limit)

        session = self.db_factory()
        try:
            return self._process_batch_internal(session, batch_limit)
        finally:
            session.close()

    def _process_batch_internal(self, db: Session, limit: int) -> BatchProcessingResult:
        pending_records = OutboxService.get_pending_events(db, limit=limit)
        if not pending_records:
            return BatchProcessingResult({"processed": 0, "delivered": 0, "failed": 0})

        logger.info(f"NotificationWorker processing batch of {len(pending_records)} outbox records")
        processed_count = 0
        failed_count = 0

        for record in pending_records:
            try:
                OutboxService.mark_processing(db, record.id)
                db.commit()

                event_dict = json.loads(record.payload_json)
                domain_event = DomainEvent(
                    event_id=event_dict.get("id", record.event_id),
                    event_type=event_dict.get("type", record.event_type),
                    schema_version=event_dict.get("version", "v1.0.0"),
                    occurred_at=event_dict.get("created_at", record.created_at.isoformat()),
                    entity_type=event_dict.get("entity_type", record.entity_type),
                    entity_id=event_dict.get("entity_id", record.entity_id),
                    organization_id=event_dict.get("organization_id", record.organization_id),
                    actor_user_id=event_dict.get("actor_user_id", record.actor_user_id),
                    correlation_id=event_dict.get("correlation_id", record.correlation_id),
                    idempotency_key=event_dict.get("idempotency_key", record.idempotency_key),
                    payload=event_dict.get("data", {})
                )

                # 1. In-App Notification Generation
                self._dispatch_in_app_notifications(db, domain_event)

                # 2. Transactional Email Dispatch
                self._dispatch_transactional_emails(db, domain_event)

                # 3. Outbound Webhook Subscriptions
                self._dispatch_webhooks(db, domain_event)

                # Mark Outbox Processed
                OutboxService.mark_processed(db, record.id)
                db.commit()
                processed_count += 1

            except Exception as exc:
                db.rollback()
                failed_count += 1
                err_msg = f"{str(exc)}\n{traceback.format_exc()}"
                logger.error(f"Error processing outbox record {record.id}: {str(exc)}", exc_info=True)
                OutboxService.mark_failed(db, record.id, error_message=err_msg)
                db.commit()

        return BatchProcessingResult({
            "processed": processed_count,
            "delivered": processed_count,
            "failed": failed_count
        })

    def _dispatch_in_app_notifications(self, db: Session, event: DomainEvent):
        """Generates in-app alerts for relevant users based on event context."""
        recipients = self._resolve_event_recipients(db, event)

        category = self._resolve_category_from_event(event.event_type)
        title = self._format_event_title(event)
        message = self._format_event_message(event)

        for user_id in recipients:
            NotificationService.create_notification(
                db=db,
                user_id=user_id,
                category=category,
                event_type=event.event_type,
                title=title,
                message=message,
                organization_id=event.organization_id,
                action_url=event.payload.get("action_url") or event.payload.get("verification_url"),
                entity_type=event.entity_type,
                entity_id=event.entity_id
            )

    def _dispatch_transactional_emails(self, db: Session, event: DomainEvent):
        """Dispatches email notifications to authorized recipients if preferences allow."""
        recipient_emails = self._resolve_recipient_emails(db, event)
        if not recipient_emails:
            return

        provider = get_email_provider()
        subject, html_body, text_body = EmailTemplateService.render(event.event_type, event.payload)

        for user_id, email in recipient_emails:
            # Check user preference
            if user_id:
                prefs = NotificationService.get_or_create_preferences(db, user_id)
                if not prefs.email_enabled and self._resolve_category_from_event(event.event_type) != NotificationCategory.SECURITY:
                    logger.info(f"Email suppressed for user {user_id}: email_enabled=False")
                    continue

            try:
                provider.send_email_sync(
                    recipient_email=email,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body
                )
                attempt = NotificationDeliveryAttempt(
                    id=str(uuid.uuid4()),
                    notification_id=None,
                    channel=NotificationChannel.EMAIL,
                    recipient=email,
                    status="SENT",
                    attempt_number=1,
                    response_code="250"
                )
                db.add(attempt)
            except Exception as exc:
                logger.error(f"Failed to dispatch email to {email}: {str(exc)}")
                attempt = NotificationDeliveryAttempt(
                    id=str(uuid.uuid4()),
                    notification_id=None,
                    channel=NotificationChannel.EMAIL,
                    recipient=email,
                    status="FAILED",
                    attempt_number=1,
                    error_message=str(exc)
                )
                db.add(attempt)

    def _dispatch_webhooks(self, db: Session, event: DomainEvent):
        """Enqueues and immediately delivers outbound webhooks for event."""
        deliveries = WebhookDeliveryService.enqueue_deliveries_for_event(db, event)
        for d in deliveries:
            WebhookDeliveryService.deliver_webhook(db, d.id)

    @staticmethod
    def _resolve_event_recipients(db: Session, event: DomainEvent) -> List[str]:
        """Resolves target user IDs for in-app alerts based on domain event metadata."""
        recipients = set()

        # Direct user association
        if event.payload.get("donor_id"):
            recipients.add(event.payload["donor_id"])
        if event.payload.get("user_id"):
            recipients.add(event.payload["user_id"])
        if event.payload.get("recipient_user_id"):
            recipients.add(event.payload["recipient_user_id"])
        if event.actor_user_id:
            recipients.add(event.actor_user_id)

        # Reviewer assignments
        if event.payload.get("reviewer_id"):
            recipients.add(event.payload["reviewer_id"])
        if event.payload.get("reviewer_ids"):
            for r_id in event.payload["reviewer_ids"]:
                recipients.add(r_id)

        # Organization owners & admins
        if event.organization_id and not recipients:
            memberships = db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == event.organization_id
                )
            ).scalars().all()
            for m in memberships:
                recipients.add(m.user_id)

        return list(recipients)

    @staticmethod
    def _resolve_recipient_emails(db: Session, event: DomainEvent) -> List[Tuple[Optional[str], str]]:
        """Resolves recipient (user_id, email) pairs for transactional email dispatch."""
        results = []

        # If direct email in payload
        if event.payload.get("donor_email"):
            results.append((event.payload.get("donor_id"), event.payload["donor_email"]))

        if event.payload.get("recipient_email"):
            results.append((event.payload.get("recipient_user_id"), event.payload["recipient_email"]))

        # Look up by user_id if email not in payload
        user_ids = set()
        if event.payload.get("donor_id") and not event.payload.get("donor_email"):
            user_ids.add(event.payload["donor_id"])
        if event.payload.get("user_id") and not event.payload.get("recipient_email"):
            user_ids.add(event.payload["user_id"])
        if event.payload.get("reviewer_id"):
            user_ids.add(event.payload["reviewer_id"])

        for uid in user_ids:
            user = db.execute(select(User).where(User.id == uid)).scalar_one_or_none()
            if user and user.email:
                results.append((user.id, user.email))

        # Organization notifications
        if event.organization_id and not results:
            memberships = db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == event.organization_id
                )
            ).scalars().all()
            for m in memberships:
                user = db.execute(select(User).where(User.id == m.user_id)).scalar_one_or_none()
                if user and user.email:
                    results.append((user.id, user.email))

        return results

    @staticmethod
    def _resolve_category_from_event(event_type: str) -> NotificationCategory:
        """Maps event types to notification categories."""
        if "user" in event_type or "account" in event_type:
            return NotificationCategory.ACCOUNT
        if "donation" in event_type:
            return NotificationCategory.DONATIONS
        if "payment" in event_type:
            return NotificationCategory.PAYMENTS
        if "project" in event_type:
            return NotificationCategory.PROJECTS
        if "milestone" in event_type:
            return NotificationCategory.MILESTONES
        if "evidence" in event_type:
            return NotificationCategory.EVIDENCE
        if "verification" in event_type:
            return NotificationCategory.VERIFICATION
        if "review" in event_type or "dispute" in event_type:
            return NotificationCategory.VERIFICATION
        if "disbursement" in event_type or "payout" in event_type:
            return NotificationCategory.DISBURSEMENTS
        if "certificate" in event_type:
            return NotificationCategory.CERTIFICATES
        if "security" in event_type or "compromise" in event_type:
            return NotificationCategory.SECURITY
        return NotificationCategory.SYSTEM

    @staticmethod
    def _format_event_title(event: DomainEvent) -> str:
        """Formats human-readable in-app notification title."""
        if event.payload.get("title"):
            return event.payload["title"]
        if event.event_type in ("donation.captured", "payment.captured"):
            return "Donation Payment Confirmed"
        type_str = event.event_type.replace(".", " ").replace("_", " ").title()
        return type_str

    @staticmethod
    def _format_event_message(event: DomainEvent) -> str:
        """Formats human-readable in-app notification message."""
        if event.payload.get("message"):
            return event.payload["message"]
        if event.event_type in ("donation.captured", "payment.captured"):
            amt = event.payload.get("formatted_amount")
            if not amt and "amount_minor_units" in event.payload:
                curr = event.payload.get("currency", "USD")
                val = event.payload["amount_minor_units"] / 100.0
                amt = f"${val:,.2f}" if curr == "USD" else f"{val:,.2f} {curr}"
            return f"Contribution of {amt or 'funds'} confirmed for {event.payload.get('project_title', 'project')}."
        if event.event_type == "certificate.issued":
            return f"Proof-of-Impact Certificate {event.payload.get('certificate_number')} has been cryptographically minted."
        if event.event_type == "certificate.revoked":
            return f"Proof-of-Impact Certificate {event.payload.get('certificate_number')} was revoked on the impact ledger."
        if event.event_type == "disbursement.completed":
            return f"Disbursement of {event.payload.get('formatted_amount')} transferred for {event.payload.get('milestone_title')}."
        return f"Event {event.event_type} occurred on {event.entity_type} {event.entity_id}."
