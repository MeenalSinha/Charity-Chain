"""
test_organizations_api.py - Integration tests for Organization Memberships & Invitations
========================================================================================
Tests invitations, scoped roles (OWNER, ADMIN, MEMBER), and last-owner protection.
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.models.users import User, Organization, OrganizationMembership, AccountStatus
from backend.app.core.constants import PlatformRole, OrganizationRole
from backend.app.core.exceptions import ForbiddenException, ValidationException
from backend.app.services.organization_service import OrganizationService
from backend.app.api.schemas import CreateInvitationRequest
from backend.app.api.v1.organizations import (
    get_my_organizations, get_organization_members,
    invite_member, accept_invitation_by_id, remove_member, update_member_role
)


@pytest.fixture
def org_setup(db_session):
    """Sets up an organization with an Owner user and a Member user."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    owner = User(
        email=f"owner_{ts}@ngo.org",
        full_name="NGO Owner",
        hashed_password="hash",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    member = User(
        email=f"member_{ts}@ngo.org",
        full_name="NGO Member",
        hashed_password="hash",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        name=f"Clean Water Org {ts}",
        country="USA"
    )
    db_session.add_all([owner, member, org])
    db_session.flush()

    mem_owner = OrganizationMembership(
        user_id=owner.id,
        organization_id=org.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    mem_member = OrganizationMembership(
        user_id=member.id,
        organization_id=org.id,
        role=OrganizationRole.MEMBER,
        is_active=True
    )
    db_session.add_all([mem_owner, mem_member])
    db_session.commit()

    return {"owner": owner, "member": member, "org": org}


def test_list_user_organizations_and_members(org_setup, db_session):
    """Verify listing organization memberships and member rosters."""
    owner = org_setup["owner"]
    org = org_setup["org"]

    my_orgs = get_my_organizations(current_user=owner, db=db_session)
    assert my_orgs.success is True
    assert len(my_orgs.data) >= 1
    assert any(o.organization_id == org.id and o.role == OrganizationRole.OWNER for o in my_orgs.data)

    members_res = get_organization_members(organization_id=org.id, current_user=owner, db=db_session)
    assert members_res.success is True
    assert len(members_res.data) == 2


def test_organization_invitation_workflow(org_setup, db_session):
    """Verify invitation creation by Owner and acceptance by invitee."""
    owner = org_setup["owner"]
    org = org_setup["org"]
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    invitee_email = f"invitee_{ts}@charity.org"

    # 1. Create User who will be invited
    invitee_user = User(
        email=invitee_email,
        full_name="Invited Worker",
        hashed_password="hash",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(invitee_user)
    db_session.commit()

    # 2. Owner invites new member
    inv_req = CreateInvitationRequest(invitee_email=invitee_email, role=OrganizationRole.ADMIN)
    inv_res = invite_member(organization_id=org.id, req=inv_req, current_user=owner, db=db_session)
    assert inv_res.success is True
    assert inv_res.data.invitee_email == invitee_email

    # 3. Invitee accepts invitation
    acc_res = accept_invitation_by_id(invitation_id=inv_res.data.id, current_user=invitee_user, db=db_session)
    assert acc_res.success is True
    assert acc_res.data.role == OrganizationRole.ADMIN


def test_last_owner_protection(org_setup, db_session):
    """Verify that removing or demoting the last Owner is strictly prevented."""
    owner = org_setup["owner"]
    org = org_setup["org"]

    # 1. Attempting to remove the sole Owner must fail
    with pytest.raises(ValidationException) as exc:
        remove_member(
            organization_id=org.id,
            user_id=owner.id,
            current_user=owner,
            db=db_session
        )
    assert "Cannot remove the last Owner" in str(exc.value)

    # 2. Attempting to demote the sole Owner to Member must fail
    from backend.app.api.schemas import UpdateMemberRoleRequest
    with pytest.raises(ValidationException) as exc2:
        update_member_role(
            organization_id=org.id,
            user_id=owner.id,
            req=UpdateMemberRoleRequest(role=OrganizationRole.MEMBER),
            current_user=owner,
            db=db_session
        )
    assert "Cannot demote the last Owner" in str(exc2.value)

