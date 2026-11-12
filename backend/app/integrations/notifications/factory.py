"""
factory.py - Notification & Email Provider Factory
==================================================
Instantiates and configures the appropriate notification provider
based on environment settings, with support for test dependency injection.
"""

from typing import Optional

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.integrations.notifications.base import NotificationProviderInterface
from backend.app.integrations.notifications.dev_sink import DevNotificationSink
from backend.app.integrations.notifications.smtp_provider import SMTPNotificationProvider

logger = get_logger("integrations.notifications.factory")

_configured_provider: Optional[NotificationProviderInterface] = None


def get_email_provider() -> NotificationProviderInterface:
    """Retrieves the active notification provider instance."""
    global _configured_provider
    if _configured_provider is not None:
        return _configured_provider

    provider_name = (settings.EMAIL_PROVIDER or "dev_sink").lower().strip()

    if provider_name == "smtp":
        logger.info(f"Initializing SMTP notification provider: host={settings.SMTP_HOST}:{settings.SMTP_PORT}")
        _configured_provider = SMTPNotificationProvider()
    else:
        logger.info("Initializing in-memory DevNotificationSink provider")
        _configured_provider = DevNotificationSink()

    return _configured_provider


def set_email_provider(provider: Optional[NotificationProviderInterface]) -> None:
    """Overrides the notification provider (primarily for automated testing)."""
    global _configured_provider
    _configured_provider = provider

