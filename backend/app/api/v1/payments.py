"""
payments.py - Payment Order Execution & Capture API Endpoints
=============================================================
Manages payment order capture, cancellation, and transaction state inspection.
"""

from typing import Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.api.schemas import (
    StandardResponse, CapturePaymentRequest, CapturePaymentResponse
)
from backend.app.core.constants import PlatformRole
from backend.app.core.exceptions import NotFoundException, ForbiddenException
from backend.app.core.permissions import get_current_user, get_optional_current_user
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import PaymentOrder, PaymentCapture, Donation
from backend.app.services.donation_service import DonationService

router = APIRouter(prefix="/payments", tags=["Payments"])
donation_service = DonationService()


@router.post("/orders/{order_identifier}/capture", response_model=StandardResponse)
def capture_payment(
    order_identifier: str,
    payload: Optional[CapturePaymentRequest] = None,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """
    Captures an approved PayPal order.
    Verifies gateway status before updating local records and funding aggregates.
    """
    order = db.execute(
        select(PaymentOrder).where(
            (PaymentOrder.id == order_identifier) | (PaymentOrder.provider_order_id == order_identifier)
        )
    ).scalar_one_or_none()

    if not order:
        raise NotFoundException(f"Payment order '{order_identifier}' not found")

    donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
    if donation and donation.donor_id:
        if not current_user:
            raise ForbiddenException("Authentication required to capture this payment order")
        is_owner = current_user.id == donation.donor_id
        is_admin = current_user.role in (
            PlatformRole.PLATFORM_ADMIN,
            PlatformRole.FINANCE_OPERATOR,
            PlatformRole.FINANCE_APPROVER
        )
        if not (is_owner or is_admin):
            raise ForbiddenException("You do not have permission to capture this payment order")

    user_id = current_user.id if current_user else None
    idempotency_key = payload.idempotency_key if payload else None

    res = donation_service.capture_donation_payment(
        db=db,
        order_identifier=order_identifier,
        user_id=user_id,
        idempotency_key=idempotency_key
    )
    return StandardResponse(data=res, message="Payment captured successfully")


@router.post("/orders/{order_identifier}/cancel", response_model=StandardResponse)
def cancel_payment_order(
    order_identifier: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """Marks an unapproved payment order and donation as cancelled."""
    order = db.execute(
        select(PaymentOrder).where(
            (PaymentOrder.id == order_identifier) | (PaymentOrder.provider_order_id == order_identifier)
        )
    ).scalar_one_or_none()

    if not order:
        raise NotFoundException(f"Payment order '{order_identifier}' not found")

    donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
    if donation and donation.donor_id:
        if not current_user:
            raise ForbiddenException("Authentication required to cancel this payment order")
        is_owner = current_user.id == donation.donor_id
        is_admin = current_user.role in (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
        if not (is_owner or is_admin):
            raise ForbiddenException("You do not have permission to cancel this payment order")

    user_id = current_user.id if current_user else None
    res = donation_service.cancel_donation(
        db=db,
        order_identifier=order_identifier,
        user_id=user_id
    )
    return StandardResponse(data=res, message="Payment order cancelled")


@router.get("/orders/{order_identifier}", response_model=StandardResponse)
def get_payment_order(
    order_identifier: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves payment order details by internal ID or provider order ID."""
    order = db.execute(
        select(PaymentOrder).where(
            (PaymentOrder.id == order_identifier) | (PaymentOrder.provider_order_id == order_identifier)
        )
    ).scalar_one_or_none()

    if not order:
        raise NotFoundException(f"Payment order '{order_identifier}' not found")

    donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
    if donation and donation.donor_id:
        if not current_user:
            raise ForbiddenException("Authentication required to view payment order details")
        is_owner = current_user.id == donation.donor_id
        is_staff = current_user.role in (
            PlatformRole.PLATFORM_ADMIN,
            PlatformRole.FINANCE_OPERATOR,
            PlatformRole.FINANCE_APPROVER,
            PlatformRole.FINANCE_AUDITOR
        )
        if not (is_owner or is_staff):
            raise ForbiddenException("You do not have permission to view this payment order")

    capture = db.execute(select(PaymentCapture).where(PaymentCapture.order_id == order.id)).scalar_one_or_none()

    return StandardResponse(
        data={
            "id": order.id,
            "donation_id": order.donation_id,
            "provider": order.provider,
            "provider_order_id": order.provider_order_id,
            "amount_minor_units": order.amount_minor_units,
            "currency": order.currency,
            "status": order.status.value,
            "approval_url": order.approval_url,
            "capture_id": capture.id if capture else None,
            "captured_at": capture.captured_at if capture else None
        }
    )
