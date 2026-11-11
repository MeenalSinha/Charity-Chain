"""
refunds.py - Donation Refund Processing API Endpoints
=====================================================
Governs full and partial donor refunds with PayPal Payments v2 execution and funding adjustments.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from backend.app.api.schemas import (
    StandardResponse, CreateRefundRequest, RefundResponse,
    PaginatedResponse, PaginationMeta
)
from backend.app.core.constants import PlatformRole, PaymentRefundStatus
from backend.app.core.exceptions import NotFoundException, ForbiddenException
from backend.app.core.permissions import get_current_user, require_platform_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import PaymentRefund, PaymentCapture, Donation, PaymentOrder
from backend.app.services.donation_service import DonationService

router = APIRouter(tags=["Refunds"])
donation_service = DonationService()


@router.post("/donations/{donation_id}/refunds", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
def request_donation_refund(
    donation_id: str,
    payload: CreateRefundRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initiates a full or partial refund against a captured donation.
    Enforces eligibility and reduces project/milestone net funding.
    """
    is_admin = current_user.platform_role == PlatformRole.PLATFORM_ADMIN
    res = donation_service.process_refund(
        db=db,
        donation_id=donation_id,
        amount_minor_units=payload.amount_minor_units,
        reason=payload.reason,
        actor_id=current_user.id,
        is_admin=is_admin
    )
    return StandardResponse(data=res, message="Refund processed successfully")


@router.get("/refunds/{refund_id}", response_model=StandardResponse)
def get_refund_details(
    refund_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves refund details by ID."""
    refund = db.execute(select(PaymentRefund).where(PaymentRefund.id == refund_id)).scalar_one_or_none()
    if not refund:
        raise NotFoundException(f"Refund '{refund_id}' not found")

    capture = db.execute(select(PaymentCapture).where(PaymentCapture.id == refund.capture_id)).scalar_one_or_none()
    order = db.execute(select(PaymentOrder).where(PaymentOrder.id == capture.order_id)).scalar_one_or_none() if capture else None
    donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none() if order else None

    # Check permission: actor must be donor or platform admin
    if donation and donation.donor_id != current_user.id and current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
        raise ForbiddenException("Access denied: You do not own this refund record.")

    return StandardResponse(
        data={
            "refund_id": refund.id,
            "donation_id": donation.id if donation else None,
            "provider_refund_id": refund.provider_refund_id,
            "amount_minor_units": refund.amount_minor_units,
            "currency": refund.currency,
            "reason": refund.reason,
            "status": refund.status.value,
            "refunded_at": refund.refunded_at,
            "created_at": refund.created_at
        }
    )


@router.get("/admin/refunds", response_model=PaginatedResponse)
def list_admin_refunds(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db)
):
    """Administrative overview of all processed and pending refunds across the platform."""
    offset = (page - 1) * page_size
    total = db.execute(select(func.count(PaymentRefund.id))).scalar() or 0
    items = db.execute(
        select(PaymentRefund).order_by(PaymentRefund.created_at.desc()).offset(offset).limit(page_size)
    ).scalars().all()

    data = [
        {
            "refund_id": r.id,
            "capture_id": r.capture_id,
            "provider_refund_id": r.provider_refund_id,
            "amount_minor_units": r.amount_minor_units,
            "currency": r.currency,
            "reason": r.reason,
            "status": r.status.value,
            "refunded_at": r.refunded_at,
            "created_at": r.created_at
        }
        for r in items
    ]

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return PaginatedResponse(
        data=data,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    )
