"""
notifications.py - In-App Notification Center & Preferences API Endpoints
==========================================================================
Provides user-isolated endpoints for listing notifications, unread counts,
marking read, archiving, and configuring notification preferences.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    StandardResponse, NotificationResponse, NotificationListResponse,
    NotificationPreferenceSchema
)
from backend.app.core.constants import NotificationCategory
from backend.app.core.permissions import get_current_user
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.services.notification_service import NotificationService

router = APIRouter(prefix="/notifications", tags=["Notification Center"])


@router.get("", response_model=StandardResponse[NotificationListResponse])
def list_notifications(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    unread_only: bool = Query(False, description="Filter for unread notifications only"),
    category: Optional[NotificationCategory] = Query(None, description="Filter by notification category"),
    include_archived: bool = Query(False, description="Include archived notifications"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists notifications for the authenticated user with optional category and read status filtering."""
    items, total = NotificationService.list_notifications(
        db=db,
        user_id=current_user.id,
        unread_only=unread_only,
        category=category,
        include_archived=include_archived,
        page=page,
        page_size=page_size
    )
    unread_count = NotificationService.get_unread_count(db, current_user.id)

    response_items = [
        NotificationResponse(
            id=n.id,
            user_id=n.user_id,
            organization_id=n.organization_id,
            category=n.category.value if hasattr(n.category, "value") else str(n.category),
            event_type=n.event_type,
            title=n.title,
            message=n.message,
            action_url=n.action_url,
            entity_type=n.entity_type,
            entity_id=n.entity_id,
            is_read=n.is_read,
            read_at=n.read_at,
            is_archived=n.is_archived,
            created_at=n.created_at
        )
        for n in items
    ]

    list_response = NotificationListResponse(
        items=response_items,
        total=total,
        unread_count=unread_count,
        page=page,
        page_size=page_size
    )
    return StandardResponse(data=list_response, message="Notifications retrieved successfully")


@router.get("/unread-count", response_model=StandardResponse[dict])
def get_unread_notification_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves unread notification count for badge rendering."""
    count = NotificationService.get_unread_count(db, current_user.id)
    return StandardResponse(data={"unread_count": count}, message="Unread count retrieved")


@router.get("/preferences", response_model=StandardResponse[NotificationPreferenceSchema])
def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves current notification preferences for authenticated user."""
    prefs = NotificationService.get_or_create_preferences(db, current_user.id)
    data = NotificationPreferenceSchema(
        email_enabled=prefs.email_enabled,
        in_app_enabled=prefs.in_app_enabled,
        donation_updates=prefs.donation_updates,
        project_updates=prefs.project_updates,
        milestone_updates=prefs.milestone_updates,
        verification_updates=prefs.verification_updates,
        certificate_notifications=prefs.certificate_notifications,
        security_alerts=prefs.security_alerts
    )
    return StandardResponse(data=data, message="Notification preferences retrieved")


@router.put("/preferences", response_model=StandardResponse[NotificationPreferenceSchema])
def update_notification_preferences(
    payload: NotificationPreferenceSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates notification preferences for authenticated user."""
    prefs = NotificationService.update_preferences(
        db=db,
        user_id=current_user.id,
        preferences_data=payload.model_dump()
    )
    data = NotificationPreferenceSchema(
        email_enabled=prefs.email_enabled,
        in_app_enabled=prefs.in_app_enabled,
        donation_updates=prefs.donation_updates,
        project_updates=prefs.project_updates,
        milestone_updates=prefs.milestone_updates,
        verification_updates=prefs.verification_updates,
        certificate_notifications=prefs.certificate_notifications,
        security_alerts=prefs.security_alerts
    )
    return StandardResponse(data=data, message="Notification preferences updated successfully")


@router.get("/{notification_id}", response_model=StandardResponse[NotificationResponse])
def get_notification_detail(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves details of a single notification with IDOR authorization verification."""
    n = NotificationService.get_notification(db, current_user.id, notification_id)
    data = NotificationResponse(
        id=n.id,
        user_id=n.user_id,
        organization_id=n.organization_id,
        category=n.category.value if hasattr(n.category, "value") else str(n.category),
        event_type=n.event_type,
        title=n.title,
        message=n.message,
        action_url=n.action_url,
        entity_type=n.entity_type,
        entity_id=n.entity_id,
        is_read=n.is_read,
        read_at=n.read_at,
        is_archived=n.is_archived,
        created_at=n.created_at
    )
    return StandardResponse(data=data, message="Notification retrieved")


@router.patch("/{notification_id}/read", response_model=StandardResponse[NotificationResponse])
def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Marks a single notification as read."""
    n = NotificationService.mark_as_read(db, current_user.id, notification_id)
    data = NotificationResponse(
        id=n.id,
        user_id=n.user_id,
        organization_id=n.organization_id,
        category=n.category.value if hasattr(n.category, "value") else str(n.category),
        event_type=n.event_type,
        title=n.title,
        message=n.message,
        action_url=n.action_url,
        entity_type=n.entity_type,
        entity_id=n.entity_id,
        is_read=n.is_read,
        read_at=n.read_at,
        is_archived=n.is_archived,
        created_at=n.created_at
    )
    return StandardResponse(data=data, message="Notification marked as read")


@router.post("/mark-all-read", response_model=StandardResponse[dict])
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Marks all active notifications as read for current user."""
    count = NotificationService.mark_all_as_read(db, current_user.id)
    return StandardResponse(data={"updated_count": count}, message=f"Marked {count} notifications as read")


@router.patch("/{notification_id}/archive", response_model=StandardResponse[NotificationResponse])
def archive_notification(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Archives a notification from the primary user inbox."""
    n = NotificationService.archive_notification(db, current_user.id, notification_id)
    data = NotificationResponse(
        id=n.id,
        user_id=n.user_id,
        organization_id=n.organization_id,
        category=n.category.value if hasattr(n.category, "value") else str(n.category),
        event_type=n.event_type,
        title=n.title,
        message=n.message,
        action_url=n.action_url,
        entity_type=n.entity_type,
        entity_id=n.entity_id,
        is_read=n.is_read,
        read_at=n.read_at,
        is_archived=n.is_archived,
        created_at=n.created_at
    )
    return StandardResponse(data=data, message="Notification archived successfully")

