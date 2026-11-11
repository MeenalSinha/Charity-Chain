"""
organization_service.py - Organization Membership & Invitation Domain Service
=============================================================================
Manages organization-scoped roles (OWNER, ADMIN, MEMBER), secure invitations,
membership updates, and enforces strict last-owner protection.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func

from backend.app.core.config import settings
from backend.app.core.constants import OrganizationRole, InvitationStatus, PlatformRole
from backend.app.core.exceptions import (
    CharityChainException, ConflictException, NotFoundException,
    ValidationException, ForbiddenException
)
from backend.app.core.logging import logger
from backend.app.core.security import (
    generate_secure_token, hash_token, normalize_email
)
from backend.app.db.models.users import (
    User, Organization, OrganizationMembership, OrganizationInvitation
)
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.integrations.notifications.dev_sink import notification_service
from backend.app.api.schemas import (
    OrganizationMembershipResponse, MemberDetailResponse,
    CreateInvitationRequest, OrganizationInvitationResponse
)


class OrganizationService:
    """Manages organization memberships, roles, and invitation workflows."""

    @staticmethod
    def get_user_organizations(db: Session, user_id: str) -> List[OrganizationMembershipResponse]:
        """Returns all active organization memberships for a user."""
        memberships = db.execute(
            select(OrganizationMembership, Organization)
            .join(Organization, OrganizationMembership.organization_id == Organization.id)
            .where(
                and_(
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).all()

        results = []
        for mem, org in memberships:
            results.append(
                OrganizationMembershipResponse(
                    id=mem.id,
                    organization_id=org.id,
                    organization_name=org.name,
                    role=mem.role,
                    created_at=mem.created_at
                )
            )
        return results

    @staticmethod
    def get_organization_members(db: Session, org_id: str, requesting_user_id: str) -> List[MemberDetailResponse]:
        """Returns all members of an organization if requester is a member or platform admin."""
        # Check requester membership
        requester_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == requesting_user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        requester = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
        is_platform_admin = requester and requester.role == PlatformRole.PLATFORM_ADMIN

        if not requester_mem and not is_platform_admin:
            raise ForbiddenException("You do not have access to this organization's members.")

        members = db.execute(
            select(OrganizationMembership, User)
            .join(User, OrganizationMembership.user_id == User.id)
            .where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).all()

        return [
            MemberDetailResponse(
                user_id=u.id,
                email=u.email,
                full_name=u.full_name,
                role=mem.role,
                joined_at=mem.created_at
            )
            for mem, u in members
        ]

    @staticmethod
    def create_invitation(
        db: Session,
        org_id: str,
        inviter_id: str,
        req: CreateInvitationRequest
    ) -> OrganizationInvitationResponse:
        """
        Creates an invitation for an invitee email to join an organization.
        Requires inviter to be OWNER or ADMIN in that organization.
        """
        # 1. Verify Inviter Permissions
        inviter_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == inviter_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        inviter = db.execute(select(User).where(User.id == inviter_id)).scalar_one_or_none()
        is_platform_admin = inviter and inviter.role == PlatformRole.PLATFORM_ADMIN

        if not is_platform_admin and (not inviter_mem or inviter_mem.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN)):
            raise ForbiddenException("Only organization Owners or Admins can issue invitations.")

        if req.role == OrganizationRole.OWNER and inviter_mem and inviter_mem.role != OrganizationRole.OWNER and not is_platform_admin:
            raise ForbiddenException("Only an existing Owner can invite a new Owner.")

        clean_email = normalize_email(req.invitee_email)
        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        # 2. Check If Invitee Is Already An Active Member
        existing_user = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        if existing_user:
            already_member = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == org_id,
                        OrganizationMembership.user_id == existing_user.id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            if already_member:
                raise ConflictException("User is already a member of this organization.")

        # 3. Invalidate Prior Pending Invitations For This Email In This Org
        now = datetime.utcnow()
        db.execute(
            select(OrganizationInvitation).where(
                and_(
                    OrganizationInvitation.organization_id == org_id,
                    OrganizationInvitation.invitee_email == clean_email,
                    OrganizationInvitation.status == InvitationStatus.PENDING
                )
            )
        )

        # 4. Generate Cryptographic Token
        raw_token = generate_secure_token(32)
        token_hash = hash_token(raw_token)
        expires = now + timedelta(days=settings.INVITATION_EXPIRE_DAYS)

        invitation = OrganizationInvitation(
            organization_id=org_id,
            inviter_id=inviter_id,
            invitee_email=clean_email,
            role=req.role,
            token_hash=token_hash,
            expires_at=expires,
            status=InvitationStatus.PENDING
        )
        db.add(invitation)
        db.flush()

        audit = AuditEvent(
            entity_name="OrganizationInvitation",
            entity_id=invitation.id,
            action="ORG_INVITATION_CREATED",
            actor_id=inviter_id,
            after_state={"organization_id": org_id, "invitee": clean_email, "role": req.role.value}
        )
        db.add(audit)
        db.commit()

        # Send invitation email
        notification_service.send_email_sync(
            recipient_email=clean_email,
            subject=f"Invitation to join {org.name} on CharityChain",
            html_body=f"<p>You have been invited to join {org.name} as {req.role.value}. Token: <b>{raw_token}</b></p>"
        )

        return OrganizationInvitationResponse(
            id=invitation.id,
            organization_id=invitation.organization_id,
            invitee_email=invitation.invitee_email,
            role=invitation.role,
            status=invitation.status,
            expires_at=invitation.expires_at,
            created_at=invitation.created_at
        )

    @staticmethod
    def accept_invitation(db: Session, token: str, user_id: str) -> OrganizationMembershipResponse:
        """Validates token, matches user email, and creates organization membership."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        t_hash = hash_token(token)
        now = datetime.utcnow()

        invitation = db.execute(
            select(OrganizationInvitation).where(
                and_(
                    OrganizationInvitation.token_hash == t_hash,
                    OrganizationInvitation.status == InvitationStatus.PENDING,
                    OrganizationInvitation.is_revoked == False
                )
            )
        ).scalar_one_or_none()

        if not invitation or invitation.expires_at < now:
            raise ValidationException("Invalid or expired invitation token.")

        # Security: Invitee email MUST match the logged-in user's email!
        if invitation.invitee_email.lower() != user.email.lower():
            raise ForbiddenException("This invitation was issued to a different email address.")

        # Check for duplicate membership
        existing_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == invitation.organization_id,
                    OrganizationMembership.user_id == user.id
                )
            )
        ).scalar_one_or_none()

        if existing_mem:
            if existing_mem.is_active:
                raise ConflictException("You are already an active member of this organization.")
            # Reactivate membership with new role
            existing_mem.is_active = True
            existing_mem.role = invitation.role
            mem = existing_mem
        else:
            mem = OrganizationMembership(
                user_id=user.id,
                organization_id=invitation.organization_id,
                role=invitation.role,
                is_active=True
            )
            db.add(mem)

        # Mark invitation as accepted
        invitation.status = InvitationStatus.ACCEPTED
        invitation.accepted_at = now
        db.flush()

        org = db.execute(select(Organization).where(Organization.id == invitation.organization_id)).scalar_one_or_none()

        audit = AuditEvent(
            entity_name="OrganizationMembership",
            entity_id=mem.id,
            action="ORG_INVITATION_ACCEPTED",
            actor_id=user.id,
            after_state={"organization_id": invitation.organization_id, "role": invitation.role.value}
        )
        db.add(audit)
        db.commit()

        return OrganizationMembershipResponse(
            id=mem.id,
            organization_id=invitation.organization_id,
            organization_name=org.name if org else "Organization",
            role=mem.role,
            created_at=mem.created_at
        )

    @staticmethod
    def reject_invitation(db: Session, token: str, user_id: str) -> bool:
        """Rejects pending invitation."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        t_hash = hash_token(token)

        invitation = db.execute(
            select(OrganizationInvitation).where(
                and_(
                    OrganizationInvitation.token_hash == t_hash,
                    OrganizationInvitation.status == InvitationStatus.PENDING
                )
            )
        ).scalar_one_or_none()

        if not invitation:
            raise ValidationException("Invalid invitation token.")

        if user and invitation.invitee_email.lower() != user.email.lower():
            raise ForbiddenException("This invitation was issued to a different email address.")

        invitation.status = InvitationStatus.REJECTED
        db.commit()
        return True

    @staticmethod
    def remove_member(db: Session, org_id: str, target_user_id: str, requesting_user_id: str) -> bool:
        """
        Removes a member from an organization.
        Enforces LAST OWNER PROTECTION to ensure an organization never has 0 owners.
        """
        requester_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == requesting_user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        requester = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
        is_platform_admin = requester and requester.role == PlatformRole.PLATFORM_ADMIN
        is_self = (target_user_id == requesting_user_id)

        target_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == target_user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        if not target_mem:
            raise NotFoundException("Member not found in this organization.")

        # Permission check: must be self, owner, or admin
        if not is_self and not is_platform_admin:
            if not requester_mem or requester_mem.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
                raise ForbiddenException("Only Owners or Admins may remove members.")
            if target_mem.role == OrganizationRole.OWNER and requester_mem.role != OrganizationRole.OWNER:
                raise ForbiddenException("Admins cannot remove Owners.")

        # LAST OWNER PROTECTION
        if target_mem.role == OrganizationRole.OWNER:
            owner_count = db.execute(
                select(func.count(OrganizationMembership.id)).where(
                    and_(
                        OrganizationMembership.organization_id == org_id,
                        OrganizationMembership.role == OrganizationRole.OWNER,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one()

            if owner_count <= 1:
                raise ValidationException("Cannot remove the last Owner of an organization. Transfer ownership first.")

        target_mem.is_active = False
        audit = AuditEvent(
            entity_name="OrganizationMembership",
            entity_id=target_mem.id,
            action="ORG_MEMBER_REMOVED",
            actor_id=requesting_user_id,
            after_state={"organization_id": org_id, "removed_user_id": target_user_id}
        )
        db.add(audit)
        db.commit()
        return True

    @staticmethod
    def update_member_role(
        db: Session,
        org_id: str,
        target_user_id: str,
        new_role: OrganizationRole,
        requesting_user_id: str
    ) -> bool:
        """
        Updates an organization member's role.
        Enforces LAST OWNER PROTECTION when demoting an owner.
        """
        requester_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == requesting_user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        requester = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
        is_platform_admin = requester and requester.role == PlatformRole.PLATFORM_ADMIN

        if not is_platform_admin and (not requester_mem or requester_mem.role != OrganizationRole.OWNER):
            raise ForbiddenException("Only organization Owners can update member roles.")

        target_mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == target_user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        if not target_mem:
            raise NotFoundException("Member not found in this organization.")

        # LAST OWNER PROTECTION when demoting an owner
        if target_mem.role == OrganizationRole.OWNER and new_role != OrganizationRole.OWNER:
            owner_count = db.execute(
                select(func.count(OrganizationMembership.id)).where(
                    and_(
                        OrganizationMembership.organization_id == org_id,
                        OrganizationMembership.role == OrganizationRole.OWNER,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one()

            if owner_count <= 1:
                raise ValidationException("Cannot demote the last Owner of an organization.")

        target_mem.role = new_role
        audit = AuditEvent(
            entity_name="OrganizationMembership",
            entity_id=target_mem.id,
            action="ORG_MEMBER_ROLE_UPDATED",
            actor_id=requesting_user_id,
            after_state={"target_user_id": target_user_id, "new_role": new_role.value}
        )
        db.add(audit)
        db.commit()
        return True

