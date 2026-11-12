"""
webhook_subscription_service.py - External Webhook Subscription Domain Service
==============================================================================
Manages webhook subscription lifecycles, SSRF URL validation, AES-256 secret encryption,
secret rotation, and tenant-isolated subscription governance.
"""

import uuid
import secrets
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.config import settings
from backend.app.core.exceptions import NotFoundException, ForbiddenException, ValidationException
from backend.app.core.logging import get_logger
from backend.app.core.ssrf_protection import validate_webhook_url, SSRFValidationError
from backend.app.db.models.notifications import WebhookSubscription, WebhookEndpointSecret

logger = get_logger("services.webhooks.subscriptions")


def _get_fernet_cipher() -> Fernet:
    """Derives a Fernet cipher for webhook secrets from ENCRYPTION_KEY or SECRET_KEY."""
    raw_key = getattr(settings, "ENCRYPTION_KEY", None) or settings.SECRET_KEY
    key_bytes = hashlib.sha256(raw_key.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key_bytes))


class WebhookSubscriptionService:
    """Domain service managing outbound webhook endpoint subscriptions and secret lifecycles."""

    @classmethod
    def create_subscription(
        cls,
        db: Session,
        created_by_user_id: str,
        endpoint_url: str,
        organization_id: Optional[str] = None,
        description: Optional[str] = None,
        subscribed_events: Optional[List[str]] = None
    ) -> Tuple[WebhookSubscription, str]:
        """
        Creates a new webhook subscription, generates an HMAC signing secret,
        encrypts the secret at rest, and returns (subscription, plaintext_secret).
        The plaintext secret is returned ONLY ONCE upon creation.
        """
        # 1. Validate destination URL against SSRF policy
        validate_webhook_url(endpoint_url)

        # 2. Normalize subscribed events
        events = subscribed_events or ["*"]
        if not isinstance(events, list):
            raise ValidationException("subscribed_events must be a list of event strings.")

        sub = WebhookSubscription(
            id=str(uuid.uuid4()),
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            endpoint_url=endpoint_url.strip(),
            description=description.strip() if description else None,
            subscribed_events=events,
            consecutive_failures=0
        )
        db.add(sub)
        db.flush()

        # 3. Generate and encrypt signing secret
        raw_secret = f"whsec_{secrets.token_urlsafe(32)}"
        cipher = _get_fernet_cipher()
        encrypted_secret = cipher.encrypt(raw_secret.encode()).decode()
        prefix = raw_secret[:12] + "..."

        secret_record = WebhookEndpointSecret(
            id=str(uuid.uuid4()),
            subscription_id=sub.id,
            secret_encrypted=encrypted_secret,
            secret_prefix=prefix
        )
        db.add(secret_record)
        db.commit()
        db.refresh(sub)

        logger.info(f"Created webhook subscription {sub.id} for org={organization_id} by user={created_by_user_id}")
        return sub, raw_secret

    @classmethod
    def rotate_secret(cls, db: Session, subscription_id: str, requesting_user_id: str) -> str:
        """
        Rotates the HMAC signing secret for a subscription.
        Deactivates prior secrets and returns the new plaintext secret once.
        """
        sub = cls.get_subscription(db, subscription_id, requesting_user_id)

        now = datetime.now(timezone.utc)
        # Deactivate existing active secrets
        active_secrets = db.execute(
            select(WebhookEndpointSecret).where(
                and_(
                    WebhookEndpointSecret.subscription_id == sub.id,
                    WebhookEndpointSecret.is_active.is_(True)
                )
            )
        ).scalars().all()

        for s in active_secrets:
            s.is_active = False
            s.rotated_at = now

        # Create new secret
        raw_secret = f"whsec_{secrets.token_urlsafe(32)}"
        cipher = _get_fernet_cipher()
        encrypted_secret = cipher.encrypt(raw_secret.encode()).decode()
        prefix = raw_secret[:12] + "..."

        new_secret_record = WebhookEndpointSecret(
            id=str(uuid.uuid4()),
            subscription_id=sub.id,
            secret_encrypted=encrypted_secret,
            secret_prefix=prefix
        )
        db.add(new_secret_record)
        db.commit()

        logger.info(f"Rotated webhook signing secret for subscription {sub.id}")
        return raw_secret

    @classmethod
    def get_subscription(cls, db: Session, subscription_id: str, requesting_user_id: Optional[str] = None) -> WebhookSubscription:
        """Retrieves a subscription by ID with authorization verification."""
        sub = db.execute(
            select(WebhookSubscription).where(WebhookSubscription.id == subscription_id)
        ).scalar_one_or_none()

        if not sub:
            raise NotFoundException(f"Webhook subscription {subscription_id} not found.")

        # If requesting_user_id provided, verify creator ownership or platform admin
        if requesting_user_id and sub.created_by_user_id != requesting_user_id:
            from backend.app.db.models.users import User
            from backend.app.core.constants import PlatformRole
            user = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
            if not user or user.role != PlatformRole.PLATFORM_ADMIN:
                raise ForbiddenException("Unauthorized access to webhook subscription.")

        return sub

    @staticmethod
    def list_subscriptions(
        db: Session,
        organization_id: Optional[str] = None,
        created_by_user_id: Optional[str] = None
    ) -> List[WebhookSubscription]:
        """Lists active and inactive subscriptions filtered by organization or creator."""
        conditions = []
        if organization_id:
            conditions.append(WebhookSubscription.organization_id == organization_id)
        if created_by_user_id:
            conditions.append(WebhookSubscription.created_by_user_id == created_by_user_id)

        query = select(WebhookSubscription).order_by(WebhookSubscription.created_at.desc())
        if conditions:
            query = query.where(and_(*conditions))

        return list(db.execute(query).scalars().all())

    @classmethod
    def update_subscription(
        cls,
        db: Session,
        subscription_id: str,
        requesting_user_id: str,
        endpoint_url: Optional[str] = None,
        description: Optional[str] = None,
        subscribed_events: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> WebhookSubscription:
        """Updates subscription attributes, revalidating endpoint URLs against SSRF."""
        sub = cls.get_subscription(db, subscription_id, requesting_user_id)

        if endpoint_url is not None:
            validate_webhook_url(endpoint_url)
            sub.endpoint_url = endpoint_url.strip()

        if description is not None:
            sub.description = description.strip() if description else None

        if subscribed_events is not None:
            if not isinstance(subscribed_events, list):
                raise ValidationException("subscribed_events must be a list of event strings.")
            sub.subscribed_events = subscribed_events

        if is_active is not None:
            sub.is_active = bool(is_active)
            if sub.is_active:
                sub.consecutive_failures = 0
                sub.suspended_at = None

        sub.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(sub)
        return sub

    @classmethod
    def delete_subscription(cls, db: Session, subscription_id: str, requesting_user_id: str) -> bool:
        """Deletes a webhook subscription and cascades deletion to secrets and deliveries."""
        sub = cls.get_subscription(db, subscription_id, requesting_user_id)
        db.delete(sub)
        db.commit()
        logger.info(f"Deleted webhook subscription {subscription_id}")
        return True

    @staticmethod
    def get_plaintext_secret(db: Session, subscription_id: str) -> Optional[str]:
        """Internal helper decrypting the active secret for signature generation."""
        secret_rec = db.execute(
            select(WebhookEndpointSecret).where(
                and_(
                    WebhookEndpointSecret.subscription_id == subscription_id,
                    WebhookEndpointSecret.is_active.is_(True)
                )
            ).order_by(WebhookEndpointSecret.created_at.desc())
        ).scalars().first()

        if not secret_rec:
            return None

        cipher = _get_fernet_cipher()
        try:
            decrypted = cipher.decrypt(secret_rec.secret_encrypted.encode()).decode("utf-8")
            return decrypted
        except Exception as exc:
            logger.error(f"Failed to decrypt secret for subscription {subscription_id}: {str(exc)}")
            return None
