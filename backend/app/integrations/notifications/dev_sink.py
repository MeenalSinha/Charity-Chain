"""
dev_sink.py - Development Notification Sink & Provider Implementation
======================================================================
Captures outgoing transactional emails in memory for development and testing.
Fails safely in production when external SMTP credentials are not configured.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.app.core.config import settings
from backend.app.core.constants import EnvironmentType
from backend.app.core.exceptions import CharityChainException
from backend.app.core.logging import logger
from backend.app.integrations.notifications.base import NotificationProviderInterface


class NotificationDeliveryException(CharityChainException):
    """Raised when notification delivery fails."""
    def __init__(self, message: str = "Failed to deliver notification.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="NOTIFICATION_DELIVERY_FAILED", status_code=503, details=details)


class DevNotificationSink(NotificationProviderInterface):
    """
    Isolated development & testing email sink.
    Captures sent emails into an in-memory inspectable log.
    In PRODUCTION, fails closed unless real SMTP is configured.
    """
    def __init__(self):
        self.outbox: List[Dict[str, Any]] = []

    def send_email_sync(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        if settings.ENVIRONMENT == EnvironmentType.PRODUCTION:
            if not settings.SMTP_HOST or settings.SMTP_HOST == "localhost":
                logger.error("Production email delivery aborted: SMTP_HOST not configured.")
                raise NotificationDeliveryException("Outbound email provider is unavailable in production.")

        entry = {
            "recipient_email": recipient_email,
            "to_email": recipient_email,
            "subject": subject,
            "html_body": html_body,
            "text_body": text_body or html_body,
            "sent_at": datetime.utcnow().isoformat(),
        }
        self.outbox.append(entry)
        logger.info(f"DevNotificationSink: Email recorded to outbox for <{recipient_email}>: '{subject}'")
        return True

    @property
    def captured_messages(self) -> List[Dict[str, Any]]:
        return self.outbox

    async def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        return self.send_email_sync(recipient_email, subject, html_body, text_body)

    async def send_in_app_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        logger.info(f"DevNotificationSink: In-app notification for user <{user_id}>: '{title}'")
        return True

    def verify_connection(self) -> bool:
        """Validates provider readiness. In production, dev sink is disallowed."""
        if settings.ENVIRONMENT == EnvironmentType.PRODUCTION:
            return False
        return True

    def get_latest_email_for(self, email: str) -> Optional[Dict[str, Any]]:
        """Helper for test assertions retrieving the most recent email to a recipient."""
        for msg in reversed(self.outbox):
            if msg["recipient_email"].lower() == email.lower():
                return msg
        return None

    def clear(self):
        """Clears the dev outbox."""
        self.outbox.clear()


# Global default instance
notification_service = DevNotificationSink()

