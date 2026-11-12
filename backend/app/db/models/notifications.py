"""
notifications.py - Notification, Outbox, and Webhook Domain Models
==================================================================
Implements durable transactional outbox messaging, in-app notifications,
user notification preferences, delivery tracking, and outbound webhook subscriptions.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, Text, Boolean, JSON,
    Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship

from backend.app.db.base import Base
from backend.app.core.constants import (
    NotificationCategory, NotificationChannel, OutboxStatus,
    WebhookDeliveryStatus, WebhookSubscriptionStatus
)


class NotificationOutbox(Base):
    """
    Database-backed transactional outbox for reliable event dispatch.
    Ensures domain state changes and event creation are committed atomically.
    """
    __tablename__ = "notification_outbox"

    event_id = Column(String(64), nullable=False, unique=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(64), nullable=False)
    organization_id = Column(String(36), nullable=True, index=True)
    actor_user_id = Column(String(36), nullable=True)
    correlation_id = Column(String(64), nullable=True)
    idempotency_key = Column(String(128), nullable=False, unique=True, index=True)
    payload_json = Column(Text, nullable=False)
    status = Column(SQLEnum(OutboxStatus), default=OutboxStatus.PENDING, nullable=False, index=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=5, nullable=False)
    next_retry_at = Column(DateTime(timezone=True), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_notification_outbox_status_next_retry", "status", "next_retry_at"),
    )

    @property
    def attempt_count(self) -> int:
        return self.retry_count

    @property
    def last_error(self) -> Optional[str]:
        return self.error_message


class Notification(Base):
    """
    In-app notification records presented in user workspaces.
    """
    __tablename__ = "notifications"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True)
    category = Column(SQLEnum(NotificationCategory), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    action_url = Column(String(512), nullable=True)
    entity_type = Column(String(64), nullable=True)
    entity_id = Column(String(64), nullable=True)
    is_read = Column(Boolean, default=False, nullable=False, index=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_notifications_user_unread", "user_id", "is_read"),
    )


class NotificationPreference(Base):
    """
    User communication channel and category preferences.
    """
    __tablename__ = "notification_preferences"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    email_enabled = Column(Boolean, default=True, nullable=False)
    in_app_enabled = Column(Boolean, default=True, nullable=False)
    donation_updates = Column(Boolean, default=True, nullable=False)
    project_updates = Column(Boolean, default=True, nullable=False)
    milestone_updates = Column(Boolean, default=True, nullable=False)
    verification_updates = Column(Boolean, default=True, nullable=False)
    certificate_notifications = Column(Boolean, default=True, nullable=False)
    security_alerts = Column(Boolean, default=True, nullable=False)  # Mandatory security invariant


class NotificationDeliveryAttempt(Base):
    """
    Audit log for each delivery attempt across notification channels.
    """
    __tablename__ = "notification_delivery_attempts"

    notification_id = Column(String(36), ForeignKey("notifications.id", ondelete="CASCADE"), nullable=True, index=True)
    channel = Column(SQLEnum(NotificationChannel), nullable=False, index=True)
    recipient = Column(String(255), nullable=False)
    status = Column(String(32), default="PENDING", nullable=False)  # PENDING, SENT, FAILED
    attempt_number = Column(Integer, default=1, nullable=False)
    response_code = Column(String(32), nullable=True)
    error_message = Column(Text, nullable=True)
    attempted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class WebhookSubscription(Base):
    """
    External webhook subscriber registration.
    Scoped to an organization or platform administrator.
    """
    __tablename__ = "webhook_subscriptions"

    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    endpoint_url = Column(String(512), nullable=False)
    description = Column(String(255), nullable=True)
    subscribed_events = Column(JSON, default=list, nullable=False)
    consecutive_failures = Column(Integer, default=0, nullable=False)
    suspended_at = Column(DateTime(timezone=True), nullable=True)

    secrets = relationship("WebhookEndpointSecret", back_populates="subscription", cascade="all, delete-orphan")
    deliveries = relationship("WebhookDelivery", back_populates="subscription", cascade="all, delete-orphan")

    @property
    def status(self) -> WebhookSubscriptionStatus:
        if not self.is_active:
            return WebhookSubscriptionStatus.SUSPENDED if self.suspended_at else WebhookSubscriptionStatus.INACTIVE
        return WebhookSubscriptionStatus.ACTIVE


class WebhookEndpointSecret(Base):
    """
    Asymmetric / symmetric HMAC signing secret for webhook verification.
    Encrypted at rest using Fernet AES-256.
    """
    __tablename__ = "webhook_endpoint_secrets"

    subscription_id = Column(String(36), ForeignKey("webhook_subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    secret_encrypted = Column(Text, nullable=False)
    secret_prefix = Column(String(16), nullable=False)
    rotated_at = Column(DateTime(timezone=True), nullable=True)

    subscription = relationship("WebhookSubscription", back_populates="secrets")

    @property
    def encrypted_secret(self) -> str:
        return self.secret_encrypted


class WebhookDelivery(Base):
    """
    Durable delivery record for an event dispatch to a specific webhook subscription.
    """
    __tablename__ = "webhook_deliveries"

    subscription_id = Column(String(36), ForeignKey("webhook_subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    payload_json = Column(Text, nullable=False)
    status = Column(SQLEnum(WebhookDeliveryStatus), default=WebhookDeliveryStatus.PENDING, nullable=False, index=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=5, nullable=False)
    next_retry_at = Column(DateTime(timezone=True), nullable=True, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    subscription = relationship("WebhookSubscription", back_populates="deliveries")
    attempts = relationship("WebhookDeliveryAttempt", back_populates="delivery", cascade="all, delete-orphan")


class WebhookDeliveryAttempt(Base):
    """
    Individual HTTP dispatch attempt log for a webhook delivery.
    """
    __tablename__ = "webhook_delivery_attempts"

    delivery_id = Column(String(36), ForeignKey("webhook_deliveries.id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_number = Column(Integer, default=1, nullable=False)
    status_code = Column(Integer, nullable=True)
    response_headers_json = Column(Text, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_duration_ms = Column(Integer, default=0, nullable=False)
    attempted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    delivery = relationship("WebhookDelivery", back_populates="attempts")


class NotificationTemplateVersion(Base):
    """
    Versioned email and notification template definitions.
    """
    __tablename__ = "notification_template_versions"

    template_key = Column(String(64), nullable=False, index=True)
    version = Column(String(32), default="v1.0.0", nullable=False)
    subject_template = Column(String(255), nullable=False)
    html_template = Column(Text, nullable=False)
    text_template = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("template_key", "version", name="uq_notification_template_key_version"),
    )
