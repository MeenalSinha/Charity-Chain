"""
notification_service.py - In-App Notification Center & Preferences Domain Service
================================================================================
Manages durable, user-isolated in-app notifications, unread tracking,
read/archive states, and user communication preferences.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func, update

from backend.app.core.constants import NotificationCategory
from backend.app.core.exceptions import NotFoundException, ForbiddenException, ValidationException
from backend.app.core.logging import get_logger
from backend.app.db.models.notifications import Notification, NotificationPreference

logger = get_logger("services.notifications")


class NotificationService:
    """Service managing in-app notifications and recipient preferences."""

    @staticmethod
    def get_or_create_preferences(db: Session, user_id: str) -> NotificationPreference:
        """Retrieves existing user preferences or seeds default settings."""
        prefs = db.execute(
            select(NotificationPreference).where(NotificationPreference.user_id == user_id)
        ).scalar_one_or_none()

        if not prefs:
            prefs = NotificationPreference(
                id=str(uuid.uuid4()),
                user_id=user_id,
                email_enabled=True,
                in_app_enabled=True,
                donation_updates=True,
                project_updates=True,
                milestone_updates=True,
                verification_updates=True,
                certificate_notifications=True,
                security_alerts=True
            )
            db.add(prefs)
            db.flush()
        return prefs

    @classmethod
    def update_preferences(
        cls,
        db: Session,
        user_id: str,
        preferences_data: Dict[str, Any]
    ) -> NotificationPreference:
        """Updates user communication preferences, strictly protecting security alert invariants."""
        prefs = cls.get_or_create_preferences(db, user_id)

        # Allowable toggle updates
        toggles = [
            "email_enabled", "in_app_enabled", "donation_updates",
            "project_updates", "milestone_updates", "verification_updates",
            "certificate_notifications"
        ]

        for key in toggles:
            if key in preferences_data and preferences_data[key] is not None:
                setattr(prefs, key, bool(preferences_data[key]))

        # Security invariant: security alerts cannot be disabled by user
        prefs.security_alerts = True
        prefs.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(prefs)
        return prefs

    @classmethod
    def create_notification(
        cls,
        db: Session,
        user_id: str,
        category: NotificationCategory | str,
        event_type: str,
        title: str,
        message: str,
        organization_id: Optional[str] = None,
        action_url: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> Optional[Notification]:
        """
        Creates an in-app notification if user preferences permit it.
        Security alerts are always delivered regardless of user preference.
        """
        category_enum = category if isinstance(category, NotificationCategory) else NotificationCategory(category)
        prefs = cls.get_or_create_preferences(db, user_id)

        # Check in-app delivery permission
        if not prefs.in_app_enabled and category_enum != NotificationCategory.SECURITY:
            logger.info(f"Notification suppressed for user {user_id}: in_app_enabled=False")
            return None

        # Check category-specific filters
        if category_enum in (NotificationCategory.DONATIONS, NotificationCategory.PAYMENTS) and not prefs.donation_updates:
            logger.info(f"Notification suppressed for user {user_id}: donation_updates=False")
            return None
        if category_enum == NotificationCategory.PROJECTS and not prefs.project_updates:
            logger.info(f"Notification suppressed for user {user_id}: project_updates=False")
            return None
        if category_enum == NotificationCategory.MILESTONES and not prefs.milestone_updates:
            logger.info(f"Notification suppressed for user {user_id}: milestone_updates=False")
            return None
        if category_enum in (NotificationCategory.EVIDENCE, NotificationCategory.VERIFICATION) and not prefs.verification_updates:
            logger.info(f"Notification suppressed for user {user_id}: verification_updates=False")
            return None
        if category_enum == NotificationCategory.CERTIFICATES and not prefs.certificate_notifications:
            logger.info(f"Notification suppressed for user {user_id}: certificate_notifications=False")
            return None

        notif = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            organization_id=organization_id,
            category=category_enum,
            event_type=event_type,
            title=title,
            message=message,
            action_url=action_url,
            entity_type=entity_type,
            entity_id=entity_id,
            is_read=False,
            is_archived=False
        )
        db.add(notif)
        db.flush()
        return notif

    create_in_app_notification = create_notification

    @staticmethod
    def list_notifications(
        db: Session,
        user_id: str,
        unread_only: bool = False,
        category: Optional[NotificationCategory] = None,
        include_archived: bool = False,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Notification], int]:
        """Lists notifications scoped strictly to the requesting user."""
        conditions = [Notification.user_id == user_id]

        if unread_only:
            conditions.append(Notification.is_read.is_(False))
        if not include_archived:
            conditions.append(Notification.is_archived.is_(False))
        if category:
            conditions.append(Notification.category == category)

        count_query = select(func.count(Notification.id)).where(and_(*conditions))
        total = db.execute(count_query).scalar_one()

        offset = max(0, (page - 1) * page_size)
        query = select(Notification).where(and_(*conditions)).order_by(
            Notification.created_at.desc()
        ).offset(offset).limit(page_size)

        notifications = list(db.execute(query).scalars().all())
        return notifications, total

    @staticmethod
    def get_unread_count(db: Session, user_id: str) -> int:
        """Returns the number of unread, non-archived notifications for user."""
        query = select(func.count(Notification.id)).where(
            and_(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
                Notification.is_archived.is_(False)
            )
        )
        return db.execute(query).scalar_one()

    @staticmethod
    def get_notification(db: Session, user_id: str, notification_id: str) -> Notification:
        """Retrieves a single notification, verifying user ownership (IDOR defense)."""
        notif = db.execute(
            select(Notification).where(Notification.id == notification_id)
        ).scalar_one_or_none()

        if not notif:
            raise NotFoundException(f"Notification {notification_id} not found.")

        if notif.user_id != user_id:
            logger.warning(f"IDOR violation attempt: user {user_id} tried accessing notification {notification_id} owned by {notif.user_id}")
            raise ForbiddenException("Access to this notification is unauthorized.")

        return notif

    @classmethod
    def mark_as_read(cls, db: Session, user_id: str, notification_id: str) -> Notification:
        """Marks a notification as read with verified ownership."""
        notif = cls.get_notification(db, user_id, notification_id)
        if not notif.is_read:
            notif.is_read = True
            notif.read_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(notif)
        return notif

    @staticmethod
    def mark_all_as_read(db: Session, user_id: str) -> int:
        """Marks all active notifications for user as read."""
        now = datetime.now(timezone.utc)
        stmt = (
            update(Notification)
            .where(and_(Notification.user_id == user_id, Notification.is_read.is_(False)))
            .values(is_read=True, read_at=now)
        )
        result = db.execute(stmt)
        db.commit()
        return result.rowcount

    @classmethod
    def archive_notification(cls, db: Session, user_id: str, notification_id: str) -> Notification:
        """Archives a notification with verified ownership."""
        notif = cls.get_notification(db, user_id, notification_id)
        if not notif.is_archived:
            notif.is_archived = True
            notif.archived_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(notif)
        return notif
