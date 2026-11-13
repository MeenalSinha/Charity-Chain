"""
donations.py - Charitable Donations API Endpoints
=================================================
Handles donation intent creation, PayPal checkout initialization, and donor transaction tracking.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from backend.app.api.schemas import (
    CreateDonationRequest, DonationResponse, StandardResponse,
    PaginatedResponse, PaginationMeta
)
from backend.app.core.constants import PlatformRole, DonationStatus
from backend.app.core.exceptions import NotFoundException, ForbiddenException
from backend.app.core.permissions import get_current_user, get_optional_current_user
from backend.app.core.rate_limiter import financial_rate_limit
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import Donation, PaymentOrder
from backend.app.services.donation_service import DonationService

router = APIRouter(prefix="/donations", tags=["Donations"])
donation_service = DonationService()


@router.post("", response_model=StandardResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(financial_rate_limit())])
def create_donation(
    payload: CreateDonationRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """
    Creates a new donation intent and initiates a genuine PayPal Orders v2 checkout order.
    Returns approval URL for redirecting the donor to PayPal.
    """
    donor_id = current_user.id if current_user else None
    donor_name = payload.donor_name or (current_user.email if current_user else "Anonymous Donor")
    donor_email = payload.donor_email or (current_user.email if current_user else None)

    res = donation_service.create_donation_intent(
        db=db,
        project_id=payload.project_id,
        milestone_id=payload.milestone_id,
        amount_minor_units=payload.amount_minor_units,
        currency=payload.currency,
        donor_id=donor_id,
        donor_name=donor_name,
        donor_email=donor_email,
        is_anonymous=payload.is_anonymous,
        return_url=payload.return_url,
        cancel_url=payload.cancel_url
    )
    return StandardResponse(data=res, message="Donation intent created successfully")


@router.get("/my", response_model=PaginatedResponse)
def get_my_donations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all donations made by the currently authenticated donor."""
    items, total = donation_service.list_donor_donations(
        db=db,
        donor_id=current_user.id,
        page=page,
        page_size=page_size
    )

    data = []
    for d in items:
        order = db.execute(select(PaymentOrder).where(PaymentOrder.donation_id == d.id)).scalar_one_or_none()
        data.append(
            DonationResponse(
                id=d.id,
                donor_id=d.donor_id,
                project_id=d.project_id,
                milestone_id=d.milestone_id,
                amount_minor_units=d.amount_minor_units,
                currency=d.currency,
                status=d.status,
                donor_name=d.donor_name if not d.is_anonymous else "Anonymous Donor",
                is_anonymous=d.is_anonymous,
                order_id=order.id if order else None,
                provider_order_id=order.provider_order_id if order else None,
                approval_url=order.approval_url if order else None,
                completed_at=d.completed_at,
                created_at=d.created_at
            ).model_dump()
        )

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


@router.get("/{donation_id}", response_model=StandardResponse)
def get_donation_by_id(
    donation_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves donation details with access control verification."""
    user_id = current_user.id if current_user else None
    donation = donation_service.get_donation(db, donation_id, user_id=user_id)
    order = db.execute(select(PaymentOrder).where(PaymentOrder.donation_id == donation.id)).scalar_one_or_none()

    resp = DonationResponse(
        id=donation.id,
        donor_id=donation.donor_id,
        project_id=donation.project_id,
        milestone_id=donation.milestone_id,
        amount_minor_units=donation.amount_minor_units,
        currency=donation.currency,
        status=donation.status,
        donor_name=donation.donor_name if not donation.is_anonymous else "Anonymous Donor",
        is_anonymous=donation.is_anonymous,
        order_id=order.id if order else None,
        provider_order_id=order.provider_order_id if order else None,
        approval_url=order.approval_url if order else None,
        completed_at=donation.completed_at,
        created_at=donation.created_at
    )
    return StandardResponse(data=resp.model_dump())
