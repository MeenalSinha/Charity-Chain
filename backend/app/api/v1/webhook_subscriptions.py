"""
webhook_subscriptions.py - Outbound Webhook Subscription & Management API
==========================================================================
Provides management endpoints for external webhook subscriptions, secret rotation,
delivery attempt inspection, manual retries, and event type catalogs.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.api.schemas import (
    StandardResponse, CreateWebhookSubscriptionRequest, UpdateWebhookSubscriptionRequest,
    WebhookSubscriptionResponse, WebhookSubscriptionCreatedResponse,
    WebhookDeliveryResponse, WebhookDeliveryAttemptResponse
)
from backend.app.core.constants import DomainEventType, PlatformRole
from backend.app.core.events import DomainEvent
from backend.app.core.permissions import get_current_user
from backend.app.db.session import get_db
from backend.app.db.models.users import User, OrganizationMembership
from backend.app.db.models.notifications import WebhookSubscription, WebhookDelivery, WebhookDeliveryAttempt
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService
from backend.app.services.webhook_delivery_service import WebhookDeliveryService

router = APIRouter(prefix="/webhooks/subscriptions", tags=["Webhook Subscriptions"])


@router.post("", response_model=StandardResponse[WebhookSubscriptionCreatedResponse], status_code=status.HTTP_201_CREATED)
def create_webhook_subscription(
    payload: CreateWebhookSubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Registers an external webhook endpoint.
    Returns the HMAC signing secret once. Store it immediately; it cannot be recovered.
    """
    # Enforce tenant scoping: non-admins must supply an organization they belong to
    if payload.organization_id:
        if current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
            membership = db.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == payload.organization_id,
                    OrganizationMembership.user_id == current_user.id
                )
            ).scalar_one_or_none()
            if not membership:
                raise HTTPException(status_code=403, detail="Not authorized to create webhooks for this organization.")

    sub, raw_secret = WebhookSubscriptionService.create_subscription(
        db=db,
        created_by_user_id=current_user.id,
        endpoint_url=payload.endpoint_url,
        organization_id=payload.organization_id,
        description=payload.description,
        subscribed_events=payload.subscribed_events
    )

    sub_response = WebhookSubscriptionResponse(
        id=sub.id,
        organization_id=sub.organization_id,
        created_by_user_id=sub.created_by_user_id,
        endpoint_url=sub.endpoint_url,
        description=sub.description,
        subscribed_events=sub.subscribed_events,
        is_active=sub.is_active,
        consecutive_failures=sub.consecutive_failures,
        suspended_at=sub.suspended_at,
        secret_prefix=sub.secrets[0].secret_prefix if sub.secrets else None,
        created_at=sub.created_at
    )

    result = WebhookSubscriptionCreatedResponse(
        subscription=sub_response,
        secret=raw_secret,
        warning="This signing secret will NEVER be displayed again. Store it securely."
    )
    return StandardResponse(data=result, message="Webhook subscription created successfully")


@router.get("", response_model=StandardResponse[List[WebhookSubscriptionResponse]])
def list_webhook_subscriptions(
    organization_id: Optional[str] = Query(None, description="Filter by organization ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists webhook subscriptions accessible to current user."""
    user_id_filter = current_user.id if current_user.platform_role != PlatformRole.PLATFORM_ADMIN else None
    subs = WebhookSubscriptionService.list_subscriptions(
        db=db,
        organization_id=organization_id,
        created_by_user_id=user_id_filter
    )

    results = []
    for s in subs:
        secret_prefix = s.secrets[0].secret_prefix if s.secrets else None
        results.append(
            WebhookSubscriptionResponse(
                id=s.id,
                organization_id=s.organization_id,
                created_by_user_id=s.created_by_user_id,
                endpoint_url=s.endpoint_url,
                description=s.description,
                subscribed_events=s.subscribed_events,
                is_active=s.is_active,
                consecutive_failures=s.consecutive_failures,
                suspended_at=s.suspended_at,
                secret_prefix=secret_prefix,
                created_at=s.created_at
            )
        )
    return StandardResponse(data=results, message="Webhook subscriptions retrieved")


@router.get("/events/types", response_model=StandardResponse[List[str]])
def list_supported_event_types():
    """Returns the catalog of all supported domain event types eligible for subscription."""
    types = [e.value for e in DomainEventType]
    return StandardResponse(data=types, message="Event types catalog retrieved")


@router.get("/{subscription_id}", response_model=StandardResponse[WebhookSubscriptionResponse])
def get_webhook_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves metadata of a specific webhook subscription (without secret)."""
    sub = WebhookSubscriptionService.get_subscription(db, subscription_id, current_user.id)
    secret_prefix = sub.secrets[0].secret_prefix if sub.secrets else None
    data = WebhookSubscriptionResponse(
        id=sub.id,
        organization_id=sub.organization_id,
        created_by_user_id=sub.created_by_user_id,
        endpoint_url=sub.endpoint_url,
        description=sub.description,
        subscribed_events=sub.subscribed_events,
        is_active=sub.is_active,
        consecutive_failures=sub.consecutive_failures,
        suspended_at=sub.suspended_at,
        secret_prefix=secret_prefix,
        created_at=sub.created_at
    )
    return StandardResponse(data=data, message="Webhook subscription retrieved")


@router.put("/{subscription_id}", response_model=StandardResponse[WebhookSubscriptionResponse])
def update_webhook_subscription(
    subscription_id: str,
    payload: UpdateWebhookSubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a webhook subscription URL, event filters, or status."""
    sub = WebhookSubscriptionService.update_subscription(
        db=db,
        subscription_id=subscription_id,
        requesting_user_id=current_user.id,
        endpoint_url=payload.endpoint_url,
        description=payload.description,
        subscribed_events=payload.subscribed_events,
        is_active=payload.is_active
    )
    secret_prefix = sub.secrets[0].secret_prefix if sub.secrets else None
    data = WebhookSubscriptionResponse(
        id=sub.id,
        organization_id=sub.organization_id,
        created_by_user_id=sub.created_by_user_id,
        endpoint_url=sub.endpoint_url,
        description=sub.description,
        subscribed_events=sub.subscribed_events,
        is_active=sub.is_active,
        consecutive_failures=sub.consecutive_failures,
        suspended_at=sub.suspended_at,
        secret_prefix=secret_prefix,
        created_at=sub.created_at
    )
    return StandardResponse(data=data, message="Webhook subscription updated successfully")


@router.delete("/{subscription_id}", response_model=StandardResponse[dict])
def delete_webhook_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes a webhook subscription."""
    WebhookSubscriptionService.delete_subscription(db, subscription_id, current_user.id)
    return StandardResponse(data={"deleted": True, "subscription_id": subscription_id}, message="Subscription deleted")


@router.post("/{subscription_id}/rotate-secret", response_model=StandardResponse[dict])
def rotate_webhook_secret(
    subscription_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rotates the HMAC signing secret. The new secret is displayed once."""
    new_secret = WebhookSubscriptionService.rotate_secret(db, subscription_id, current_user.id)
    return StandardResponse(
        data={"secret": new_secret, "warning": "Store this new signing secret immediately."},
        message="Signing secret rotated successfully"
    )


@router.post("/{subscription_id}/test", response_model=StandardResponse[dict])
def test_webhook_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dispatches a test ping event to verify subscriber endpoint connectivity."""
    sub = WebhookSubscriptionService.get_subscription(db, subscription_id, current_user.id)

    test_event = DomainEvent.create(
        event_type="test.ping",
        entity_type="webhook_subscription",
        entity_id=sub.id,
        payload={"message": "CharityChain Webhook Connectivity Test", "subscription_id": sub.id},
        organization_id=sub.organization_id,
        actor_user_id=current_user.id
    )

    deliveries = WebhookDeliveryService.enqueue_deliveries_for_event(db, test_event)
    success = False
    for d in deliveries:
        if d.subscription_id == sub.id:
            success = WebhookDeliveryService.deliver_webhook(db, d.id)
            break

    return StandardResponse(
        data={"dispatched": True, "success": success, "subscription_id": sub.id},
        message="Test event dispatched" if success else "Test event dispatched but endpoint failed or returned non-2xx"
    )


@router.get("/{subscription_id}/deliveries", response_model=StandardResponse[List[WebhookDeliveryResponse]])
def get_subscription_deliveries(
    subscription_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists recent delivery records and statuses for a webhook subscription."""
    sub = WebhookSubscriptionService.get_subscription(db, subscription_id, current_user.id)

    query = select(WebhookDelivery).where(
        WebhookDelivery.subscription_id == sub.id
    ).order_by(WebhookDelivery.created_at.desc()).limit(limit)

    deliveries = list(db.execute(query).scalars().all())
    data = [
        WebhookDeliveryResponse(
            id=d.id,
            subscription_id=d.subscription_id,
            event_id=d.event_id,
            event_type=d.event_type,
            status=d.status.value if hasattr(d.status, "value") else str(d.status),
            retry_count=d.retry_count,
            created_at=d.created_at,
            completed_at=d.completed_at
        )
        for d in deliveries
    ]
    return StandardResponse(data=data, message="Deliveries retrieved")


@router.post("/deliveries/{delivery_id}/retry", response_model=StandardResponse[dict])
def retry_webhook_delivery(
    delivery_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Manually re-triggers a delivery attempt for a failed or dead-lettered webhook dispatch."""
    success = WebhookDeliveryService.manual_retry(db, delivery_id, current_user.id)
    return StandardResponse(
        data={"delivery_id": delivery_id, "success": success},
        message="Delivery retry executed successfully" if success else "Delivery retry attempt failed"
    )

