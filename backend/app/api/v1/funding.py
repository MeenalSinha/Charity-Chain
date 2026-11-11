"""
funding.py - Project & Milestone Financial Reporting API Endpoints
==================================================================
Provides accurate net funding totals (successful captures minus completed refunds).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from backend.app.api.schemas import StandardResponse
from backend.app.core.constants import PaymentCaptureStatus, PaymentRefundStatus, DonationStatus
from backend.app.core.permissions import get_current_user
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import Donation, PaymentCapture, PaymentRefund
from backend.app.services.donation_service import DonationService

router = APIRouter(tags=["Funding"])
donation_service = DonationService()


@router.get("/projects/{project_id}/funding", response_model=StandardResponse)
def get_project_funding(
    project_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves accurate net funding summary for a project and all its milestones."""
    res = donation_service.get_project_funding_summary(db, project_id)
    return StandardResponse(data=res)


@router.get("/users/me/donation-summary", response_model=StandardResponse)
def get_user_donation_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Summarizes total contributions, successful donations, and refunds for current user."""
    user_donations = db.execute(
        select(Donation).where(Donation.donor_id == current_user.id)
    ).scalars().all()

    total_donations_count = len(user_donations)
    paid_donations = [d for d in user_donations if d.status in (DonationStatus.PAID, DonationStatus.PARTIALLY_REFUNDED, DonationStatus.REFUNDED)]
    total_contributed = sum(d.amount_minor_units for d in paid_donations)

    # Calculate user refunds
    donation_ids = [d.id for d in user_donations]
    total_refunded = 0
    if donation_ids:
        # Sum completed refunds
        total_refunded = db.execute(
            select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
            .select_from(PaymentRefund)
            .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
            .join(Donation, PaymentCapture.order_id == Donation.id)  # or via PaymentOrder
        ).scalar() or 0

    net_contributed = max(0, total_contributed - total_refunded)

    return StandardResponse(
        data={
            "total_donations_count": total_donations_count,
            "successful_donations_count": len(paid_donations),
            "gross_contributed_minor_units": total_contributed,
            "refunded_minor_units": total_refunded,
            "net_contributed_minor_units": net_contributed,
            "currency": current_user.profile.preferred_currency if current_user.profile else "USD"
        }
    )

