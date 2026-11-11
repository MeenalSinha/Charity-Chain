"""
test_privilege_escalation.py - Security tests for Privilege Escalation & Cross-Tenant Defenses
=============================================================================================
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.models.users import User, Organization, OrganizationMembership, AccountStatus
from backend.app.core.constants import PlatformRole, OrganizationRole
from backend.app.core.exceptions import ForbiddenException
from backend.app.services.auth_service import AuthService
from backend.app.services.organization_service import OrganizationService
from backend.app.api.schemas import RegisterRequest, CreateInvitationRequest
from backend.app.api.v1.auth import register
from backend.app.api.v1.organizations import invite_member, accept_invitation_by_id


def test_registration_prevents_role_injection(db_session):
    """Verify that registration never accepts or assigns elevated platform roles."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Even if someone attempts to pass role kwargs to schema or service, default DONOR is enforced
    reg_req = RegisterRequest(
        full_name="Attacker",
        email=f"attacker_{ts}@evil.org",
        password="ValidPassword123!",
        password_confirm="ValidPassword123!"
    )
    res = register(reg_req, db=db_session)
    assert res.data.role == PlatformRole.DONOR


def test_member_cannot_invite_members(db_session):
    """Verify organization MEMBER cannot issue invitations (requires OWNER or ADMIN)."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    member_user = User(
        email=f"member_{ts}@ngo.org",
        full_name="Standard Member",
        hashed_password="hash",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(name=f"Org Test {ts}", country="USA")
    db_session.add_all([member_user, org])
    db_session.flush()

    mem = OrganizationMembership(
        user_id=member_user.id,
        organization_id=org.id,
        role=OrganizationRole.MEMBER,
        is_active=True
    )
    db_session.add(mem)
    db_session.commit()

    inv_req = CreateInvitationRequest(invitee_email="target@ngo.org", role=OrganizationRole.MEMBER)
    with pytest.raises(ForbiddenException) as exc:
        invite_member(organization_id=org.id, req=inv_req, current_user=member_user, db=db_session)
    assert "Only organization Owners or Admins" in str(exc.value)


def test_cannot_accept_invitation_issued_to_different_email(db_session):
    """Verify user cannot hijack an invitation addressed to another recipient."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    owner = User(email=f"owner_{ts}@ngo.org", full_name="Owner", hashed_password="hash", role=PlatformRole.DONOR, account_status=AccountStatus.ACTIVE, is_verified=True)
    legit_invitee = User(email=f"victim_{ts}@target.org", full_name="Victim", hashed_password="hash", role=PlatformRole.DONOR, account_status=AccountStatus.ACTIVE, is_verified=True)
    attacker = User(email=f"attacker_{ts}@target.org", full_name="Attacker", hashed_password="hash", role=PlatformRole.DONOR, account_status=AccountStatus.ACTIVE, is_verified=True)
    org = Organization(name=f"Org Secure {ts}", country="USA")
    db_session.add_all([owner, legit_invitee, attacker, org])
    db_session.flush()

    mem_owner = OrganizationMembership(user_id=owner.id, organization_id=org.id, role=OrganizationRole.OWNER, is_active=True)
    db_session.add(mem_owner)
    db_session.commit()

    # Owner invites legit_invitee
    inv = OrganizationService.create_invitation(
        db_session,
        org.id,
        owner.id,
        CreateInvitationRequest(invitee_email=legit_invitee.email, role=OrganizationRole.ADMIN)
    )

    # Attacker tries to accept legit_invitee's invitation
    with pytest.raises(ForbiddenException) as exc:
        accept_invitation_by_id(invitation_id=inv.id, current_user=attacker, db=db_session)
    assert "issued to a different email" in str(exc.value)

