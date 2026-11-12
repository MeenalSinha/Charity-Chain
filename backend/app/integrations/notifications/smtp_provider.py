"""
smtp_provider.py - Standard SMTP Transactional Email Delivery Adapter
======================================================================
Production email provider utilizing standard SMTP with STARTTLS / SSL,
credential authentication, MIME multipart (HTML & plain-text) encoding,
and granular provider error handling.
"""

import smtplib
import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.integrations.notifications.base import NotificationProviderInterface
from backend.app.integrations.notifications.dev_sink import NotificationDeliveryException

logger = get_logger("integrations.notifications.smtp")


class SMTPNotificationProvider(NotificationProviderInterface):
    """Production SMTP adapter for transactional email dispatch."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: Optional[bool] = None,
        use_ssl: Optional[bool] = None,
        timeout: Optional[float] = None,
        from_address: Optional[str] = None,
        from_name: Optional[str] = None
    ):
        self.host = host or settings.SMTP_HOST
        self.port = port or settings.SMTP_PORT
        self.username = username or settings.SMTP_USERNAME
        self.password = password or settings.SMTP_PASSWORD
        self.use_tls = use_tls if use_tls is not None else settings.SMTP_USE_TLS
        self.use_ssl = use_ssl if use_ssl is not None else settings.SMTP_USE_SSL
        self.timeout = timeout if timeout is not None else settings.SMTP_TIMEOUT_SECONDS
        self.from_address = from_address or settings.EMAIL_FROM_ADDRESS
        self.from_name = from_name or settings.EMAIL_FROM_NAME

    def _create_smtp_client(self):
        """Creates and configures an authenticated SMTP connection."""
        if not self.host:
            raise NotificationDeliveryException("SMTP_HOST is not configured.")

        if self.use_ssl:
            client = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout)
        else:
            client = smtplib.SMTP(self.host, self.port, timeout=self.timeout)

        client.ehlo()

        if self.use_tls and not self.use_ssl:
            client.starttls()
            client.ehlo()

        if self.username and self.password:
            client.login(self.username, self.password)

        return client

    def verify_connection(self) -> bool:
        """Validates network connectivity and credentials with SMTP server."""
        try:
            client = self._create_smtp_client()
            client.quit()
            return True
        except Exception as exc:
            logger.error(f"SMTP connection verification failed: {str(exc)}", exc_info=True)
            return False

    def send_email_sync(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """Dispatches email message synchronously via SMTP."""
        if not recipient_email or "@" not in recipient_email:
            raise NotificationDeliveryException(f"Invalid recipient email: {recipient_email}")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.from_name} <{self.from_address}>"
        msg["To"] = recipient_email

        # Attach plain-text fallback first
        plain_text = text_body or html_body
        msg.attach(MIMEText(plain_text, "plain", "utf-8"))

        # Attach HTML body
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        client = None
        try:
            client = self._create_smtp_client()
            client.sendmail(self.from_address, [recipient_email], msg.as_string())
            logger.info(f"SMTP email successfully sent to <{recipient_email}>: '{subject}'")
            return True
        except smtplib.SMTPAuthenticationError as exc:
            logger.error(f"SMTP authentication rejected: {str(exc)}")
            raise NotificationDeliveryException(f"SMTP authentication failed: {str(exc)}")
        except smtplib.SMTPRecipientsRefused as exc:
            logger.error(f"SMTP recipient refused <{recipient_email}>: {str(exc)}")
            raise NotificationDeliveryException(f"Recipient refused: {str(exc)}")
        except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, TimeoutError) as exc:
            logger.error(f"SMTP connection error: {str(exc)}")
            raise NotificationDeliveryException(f"SMTP server unreachable: {str(exc)}")
        except Exception as exc:
            logger.error(f"Unexpected error sending email to <{recipient_email}>: {str(exc)}", exc_info=True)
            raise NotificationDeliveryException(f"Email delivery error: {str(exc)}")
        finally:
            if client:
                try:
                    client.quit()
                except Exception:
                    pass

    async def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """Asynchronously dispatches email by executing sync SMTP send in default threadpool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.send_email_sync,
            recipient_email,
            subject,
            html_body,
            text_body
        )

    async def send_in_app_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        return True

