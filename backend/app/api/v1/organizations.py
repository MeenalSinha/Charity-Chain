"""
organizations.py - Organization Memberships & Invitations API Endpoints
========================================================================
Routes for managing organization membership, issuing invitations, and updating roles.
"""

from typing import List, Optional, Any, Union
from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.constants import OrganizationRole, InvitationStatus, OrganizationDocumentType
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user, get_current_user_optional
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User, OrganizationInvitation
from backend.app.services.organization_service import OrganizationService
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.api.schemas import (
    StandardResponse, OrganizationMembershipResponse,
    MemberDetailResponse, CreateInvitationRequest,
    OrganizationInvitationResponse, UpdateMemberRoleRequest,
    CreateOrganizationRequest, UpdateOrganizationRequest,
    OrganizationDetailResponse, OrganizationPublicResponse,
    OrganizationDocumentResponse, CreateProjectRequest, ProjectResponse
)

router = APIRouter(prefix="/organizations", tags=["Organization & Onboarding Management"])


@router.post("", response_model=StandardResponse[OrganizationDetailResponse], status_code=status.HTTP_201_CREATED)
def create_organization(
    req: CreateOrganizationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Onboards a new organization and assigns caller as initial OWNER
    in an atomic database transaction.
    """
    data = NGOOnboardingService.create_organization(db, current_user.id, req)
    return StandardResponse(
        data=data,
        message=f"Organization '{data.name}' successfully registered with initial OWNER role.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/me", response_model=StandardResponse[List[OrganizationMembershipResponse]])
def get_my_organizations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists all organizations the current authenticated user belongs to."""
    data = OrganizationService.get_user_organizations(db, current_user.id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.get("/{organization_id}", response_model=StandardResponse[Any])
def get_organization(
    organization_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db_session)
):
    """
    Retrieves organization details.
    Returns confidential/legal information to members/reviewers;
    Returns public non-sensitive profile to general visitors.
    """
    data = NGOOnboardingService.get_organization(db, organization_id, current_user.id if current_user else None)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/{organization_id}", response_model=StandardResponse[OrganizationDetailResponse])
def update_organization(
    organization_id: str,
    req: UpdateOrganizationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates organization profile. Requires OWNER or ADMIN role."""
    data = NGOOnboardingService.update_organization(db, organization_id, current_user.id, req)
    return StandardResponse(data=data, message="Organization profile updated.", correlation_id=correlation_id_ctx.get())


@router.post("/{organization_id}/submit-verification", response_model=StandardResponse[OrganizationDetailResponse])
def submit_organization_verification(
    organization_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Submits organization for reviewer verification. Requires registration number and documents."""
    data = NGOOnboardingService.submit_verification(db, organization_id, current_user.id)
    return StandardResponse(
        data=data,
        message="Organization verification application submitted for reviewer examination.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/{organization_id}/documents", response_model=StandardResponse[List[OrganizationDocumentResponse]])
def list_organization_documents(
    organization_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists verification documents. Restricted to organization members or reviewers."""
    data = NGOOnboardingService.list_documents(db, organization_id, current_user.id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post("/{organization_id}/documents", response_model=StandardResponse[OrganizationDocumentResponse], status_code=status.HTTP_201_CREATED)
async def upload_organization_document(
    organization_id: str,
    document_type: OrganizationDocumentType = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Uploads a verification document. Validates MIME type, magic bytes, and file size."""
    file_bytes = await file.read()
    data = NGOOnboardingService.upload_document(
        db=db,
        org_id=organization_id,
        user_id=current_user.id,
        file_bytes=file_bytes,
        filename=file.filename or "document.pdf",
        mime_type=file.content_type or "application/pdf",
        document_type=document_type
    )
    return StandardResponse(
        data=data,
        message=f"Document '{file.filename}' uploaded successfully.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/{organization_id}/projects", response_model=StandardResponse[ProjectResponse], status_code=status.HTTP_201_CREATED)
def create_organization_project(
    organization_id: str,
    req: CreateProjectRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Creates a new charitable project for an organization. Requires OWNER or ADMIN role."""
    data = ProjectService.create_project(db, organization_id, current_user.id, req)
    return StandardResponse(
        data=data,
        message=f"Project '{data.title}' created in DRAFT status.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/{organization_id}/projects", response_model=StandardResponse[List[ProjectResponse]])
def list_organization_projects(
    organization_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db_session)
):
    """Lists projects belonging to an organization. Active/public to donors; full list to members."""
    data = ProjectService.list_organization_projects(db, organization_id, current_user.id if current_user else None)
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
    """Accepts an organization invitation with cryptographic or session verification."""
    if token:
        data = OrganizationService.accept_invitation(db, token, current_user.id)
        return StandardResponse(data=data, message="Invitation accepted. You are now a member.", correlation_id=correlation_id_ctx.get())

    # Secure verification for in-app acceptance by authenticated invitee
    from datetime import datetime
    from backend.app.core.exceptions import ForbiddenException, ValidationException, ConflictException
    from backend.app.db.models.users import OrganizationMembership, Organization
    from backend.app.db.models.reviews_impact import AuditEvent

    inv = db.execute(select(OrganizationInvitation).where(OrganizationInvitation.id == invitation_id)).scalar_one_or_none()
    if not inv:
        raise NotFoundException("Invitation not found.")

    now = datetime.utcnow()
    if inv.status != InvitationStatus.PENDING or inv.is_revoked or inv.expires_at < now:
        raise ValidationException("Invalid, expired, or already processed invitation.")

    if inv.invitee_email.lower() != current_user.email.lower():
        raise ForbiddenException("This invitation was issued to a different email address.")

    # Check for existing membership to prevent duplicate violations
    existing_mem = db.execute(
        select(OrganizationMembership).where(
            and_(
                OrganizationMembership.organization_id == inv.organization_id,
                OrganizationMembership.user_id == current_user.id
            )
        )
    ).scalar_one_or_none()

    if existing_mem:
        if existing_mem.is_active:
            raise ConflictException("You are already an active member of this organization.")
        existing_mem.is_active = True
        existing_mem.role = inv.role
        mem = existing_mem
    else:
        mem = OrganizationMembership(
            user_id=current_user.id,
            organization_id=inv.organization_id,
            role=inv.role,
            is_active=True
        )
        db.add(mem)

    inv.status = InvitationStatus.ACCEPTED
    inv.accepted_at = now
    db.flush()

    audit = AuditEvent(
        entity_name="OrganizationMembership",
        entity_id=mem.id,
        action="ORG_INVITATION_ACCEPTED",
        actor_id=current_user.id,
        after_state={"organization_id": inv.organization_id, "role": inv.role.value, "invitation_id": inv.id}
    )
    db.add(audit)
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

