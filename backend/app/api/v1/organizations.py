"""
organizations.py - Organization Memberships & Invitations API Endpoints
========================================================================
Routes for managing organization membership, issuing invitations, and updating roles.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.constants import OrganizationRole, InvitationStatus
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User, OrganizationInvitation
from backend.app.services.organization_service import OrganizationService
from backend.app.api.schemas import (
    StandardResponse, OrganizationMembershipResponse,
    MemberDetailResponse, CreateInvitationRequest,
    OrganizationInvitationResponse, UpdateMemberRoleRequest
)

router = APIRouter(prefix="/organizations", tags=["Organization Memberships & Invitations"])


@router.get("/me", response_model=StandardResponse[List[OrganizationMembershipResponse]])
def get_my_organizations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists all organizations the current authenticated user belongs to."""
    data = OrganizationService.get_user_organizations(db, current_user.id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.get("/{organization_id}/members", response_model=StandardResponse[List[MemberDetailResponse]])
def get_organization_members(
    organization_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists all members of an organization."""
    data = OrganizationService.get_organization_members(db, organization_id, current_user.id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post("/{organization_id}/invitations", response_model=StandardResponse[OrganizationInvitationResponse], status_code=status.HTTP_201_CREATED)
def invite_member(
    organization_id: str,
    req: CreateInvitationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Issues an invitation to join an organization. Requires Owner or Admin role."""
    data = OrganizationService.create_invitation(db, organization_id, current_user.id, req)
    return StandardResponse(
        data=data,
        message=f"Invitation sent to {req.invitee_email}.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/invitations", response_model=StandardResponse[List[OrganizationInvitationResponse]])
def list_my_invitations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists pending invitations sent to the current user's email address."""
    invitations = db.execute(
        select(OrganizationInvitation).where(
            and_(
                OrganizationInvitation.invitee_email == current_user.email,
                OrganizationInvitation.status == InvitationStatus.PENDING,
                OrganizationInvitation.is_revoked == False
            )
        )
    ).scalars().all()

    data = [
        OrganizationInvitationResponse(
            id=inv.id,
            organization_id=inv.organization_id,
            invitee_email=inv.invitee_email,
            role=inv.role,
            status=inv.status,
            expires_at=inv.expires_at,
            created_at=inv.created_at
        )
        for inv in invitations
    ]
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post("/invitations/{invitation_id}/accept", response_model=StandardResponse[OrganizationMembershipResponse])
def accept_invitation_by_id(
    invitation_id: str,
    token: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Accepts an organization invitation. Can pass token or invitation_id."""
    inv = db.execute(select(OrganizationInvitation).where(OrganizationInvitation.id == invitation_id)).scalar_one_or_none()
    if not inv:
        raise NotFoundException("Invitation not found.")

    if token:
        data = OrganizationService.accept_invitation(db, token, current_user.id)
    else:
        # Accept directly if logged in user email matches invitee
        if inv.invitee_email.lower() != current_user.email.lower():
            from backend.app.core.exceptions import ForbiddenException
            raise ForbiddenException("This invitation was issued to a different email address.")
        # Create membership
        from backend.app.db.models.users import OrganizationMembership, Organization
        mem = OrganizationMembership(
            user_id=current_user.id,
            organization_id=inv.organization_id,
            role=inv.role,
            is_active=True
        )
        db.add(mem)
        inv.status = InvitationStatus.ACCEPTED
        from datetime import datetime
        inv.accepted_at = datetime.utcnow()
        db.commit()

        org = db.execute(select(Organization).where(Organization.id == inv.organization_id)).scalar_one_or_none()
        data = OrganizationMembershipResponse(
            id=mem.id,
            organization_id=inv.organization_id,
            organization_name=org.name if org else "Organization",
            role=mem.role,
            created_at=mem.created_at
        )

    return StandardResponse(data=data, message="Invitation accepted. You are now a member.", correlation_id=correlation_id_ctx.get())


@router.post("/invitations/{invitation_id}/reject", response_model=StandardResponse[dict])
def reject_invitation_by_id(
    invitation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Rejects an invitation."""
    inv = db.execute(select(OrganizationInvitation).where(OrganizationInvitation.id == invitation_id)).scalar_one_or_none()
    if not inv:
        raise NotFoundException("Invitation not found.")
    if inv.invitee_email.lower() != current_user.email.lower():
        from backend.app.core.exceptions import ForbiddenException
        raise ForbiddenException("This invitation was issued to a different email address.")

    inv.status = InvitationStatus.REJECTED
    db.commit()
    return StandardResponse(data={"rejected": True}, message="Invitation rejected.", correlation_id=correlation_id_ctx.get())


@router.delete("/{organization_id}/members/{user_id}", response_model=StandardResponse[dict])
def remove_member(
    organization_id: str,
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Removes a member from an organization. Enforces last-owner protection."""
    OrganizationService.remove_member(db, organization_id, user_id, current_user.id)
    return StandardResponse(data={"removed": True}, message="Member removed.", correlation_id=correlation_id_ctx.get())


@router.patch("/{organization_id}/members/{user_id}/role", response_model=StandardResponse[dict])
def update_member_role(
    organization_id: str,
    user_id: str,
    req: UpdateMemberRoleRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates member role. Enforces last-owner protection against demotion."""
    OrganizationService.update_member_role(db, organization_id, user_id, req.role, current_user.id)
    return StandardResponse(data={"updated": True}, message="Member role updated.", correlation_id=correlation_id_ctx.get())

