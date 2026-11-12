"""
payout_recipients.py - NGO Payout Recipient Onboarding & Verification API Endpoints
==================================================================================
Manages registration, verification, primary destination setting, and status tracking
for NGO payout accounts.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    StandardResponse, RecipientCreateRequest, RecipientVerifyRequest, RecipientResponse
)
from backend.app.core.constants import PlatformRole, OrganizationRole
from backend.app.core.exceptions import ForbiddenException
from backend.app.core.permissions import get_current_user, require_platform_role, require_org_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.services.recipient_service import PayoutRecipientService

router = APIRouter(prefix="/recipients", tags=["Payout Recipients"])
recipient_service = PayoutRecipientService()


@router.post("", response_model=StandardResponse[RecipientResponse], status_code=status.HTTP_201_CREATED)
def register_payout_recipient(
    payload: RecipientCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Registers a new payout destination (e.g. PayPal email) for an organization."""
    # Check org admin/owner or platform admin
    if current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
        require_org_role(payload.organization_id, current_user.id, db, [OrganizationRole.OWNER, OrganizationRole.ADMIN])

    recipient = recipient_service.register_recipient(
        db=db,
        organization_id=payload.organization_id,
        destination=payload.destination,
        recipient_type=payload.recipient_type,
        payout_method=payload.payout_method,
        is_primary=payload.is_primary or False,
        actor_id=current_user.id
    )

    return StandardResponse(
        data=RecipientResponse(
            id=recipient.id,
            organization_id=recipient.organization_id,
            recipient_type=recipient.recipient_type.value,
            payout_method=recipient.payout_method.value,
            destination_masked=recipient.destination_masked,
            status=recipient.status.value,
            is_primary=recipient.is_primary,
            verified_at=recipient.verified_at,
            created_at=recipient.created_at
        ),
        message="Payout recipient registered and queued for verification"
    )


@router.get("/organizations/{org_id}", response_model=StandardResponse[List[RecipientResponse]])
def get_organization_recipients(
    org_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all configured payout recipients for an organization."""
    if current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
        require_org_role(org_id, current_user.id, db, [OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER])

    recipients = recipient_service.get_organization_recipients(db=db, organization_id=org_id)

    return StandardResponse(
        data=[
            RecipientResponse(
                id=r.id,
                organization_id=r.organization_id,
                recipient_type=r.recipient_type.value,
                payout_method=r.payout_method.value,
                destination_masked=r.destination_masked,
                status=r.status.value,
                is_primary=r.is_primary,
                verified_at=r.verified_at,
                created_at=r.created_at
            )
            for r in recipients
        ]
    )


@router.post("/{recipient_id}/verify", response_model=StandardResponse[RecipientResponse])
def verify_payout_recipient(
    recipient_id: str,
    payload: Optional[RecipientVerifyRequest] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Authorizes and verifies a payout recipient.
    Restricted to Platform Admin or Finance Operator.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance operators can verify payout destinations.")

    notes = payload.verification_notes if payload else None
    recipient = recipient_service.verify_recipient(
        db=db,
        recipient_id=recipient_id,
        verifier_id=current_user.id,
        verification_notes=notes
    )

    return StandardResponse(
        data=RecipientResponse(
            id=recipient.id,
            organization_id=recipient.organization_id,
            recipient_type=recipient.recipient_type.value,
            payout_method=recipient.payout_method.value,
            destination_masked=recipient.destination_masked,
            status=recipient.status.value,
            is_primary=recipient.is_primary,
            verified_at=recipient.verified_at,
            created_at=recipient.created_at
        ),
        message="Payout recipient verified successfully"
    )


@router.post("/{recipient_id}/primary", response_model=StandardResponse[RecipientResponse])
def set_primary_payout_recipient(
    recipient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Sets a verified recipient as the primary destination for its organization."""
    from backend.app.db.models.financial import PayoutRecipient
    from backend.app.core.exceptions import NotFoundException
    from sqlalchemy import select

    r = db.execute(select(PayoutRecipient).where(PayoutRecipient.id == recipient_id)).scalar_one_or_none()
    if not r:
        raise NotFoundException(f"Recipient '{recipient_id}' not found.")

    if current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
        require_org_role(r.organization_id, current_user.id, db, [OrganizationRole.OWNER, OrganizationRole.ADMIN])

    updated = recipient_service.set_primary_recipient(db=db, organization_id=r.organization_id, recipient_id=recipient_id)

    return StandardResponse(
        data=RecipientResponse(
            id=updated.id,
            organization_id=updated.organization_id,
            recipient_type=updated.recipient_type.value,
            payout_method=updated.payout_method.value,
            destination_masked=updated.destination_masked,
            status=updated.status.value,
            is_primary=updated.is_primary,
            verified_at=updated.verified_at,
            created_at=updated.created_at
        ),
        message="Primary payout recipient updated"
    )
