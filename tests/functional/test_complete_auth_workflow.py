"""
test_complete_auth_workflow.py - Functional E2E test for the complete Phase 2 Identity Lifecycle
==============================================================================================
Tests: Register -> Verify -> Login -> Profile -> Invite -> Accept -> Refresh -> Logout
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.models.users import User, Organization, OrganizationMembership, AccountStatus
from backend.app.core.constants import PlatformRole, OrganizationRole
from backend.app.integrations.notifications.dev_sink import notification_service
from backend.app.api.schemas import (
    RegisterRequest, VerifyEmailRequest, LoginRequest,
    UpdateProfileRequest, CreateInvitationRequest, RefreshTokenRequest,
    ChangePasswordRequest
)
from backend.app.api.v1.auth import (
    register, verify_email, login, refresh_token, logout, change_password, get_me
)
from backend.app.api.v1.users import update_current_user_profile, get_current_user_account
from backend.app.api.v1.organizations import invite_member, accept_invitation_by_id, get_my_organizations


class MockRequest:
    def __init__(self, host="127.0.0.1", headers=None):
        class Client:
            def __init__(self, host):
                self.host = host
        self.client = Client(host)
        self.headers = headers or {}


def test_complete_end_to_end_identity_lifecycle(db_session):
    """Executes the full real-world authenticated lifecycle from registration to logout."""
    notification_service.clear()
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    email = f"e2e_user_{ts}@charitychain.org"
    password = "InitialPassword123!"

    # 1. Registration
    reg_res = register(
        RegisterRequest(
            full_name="E2E Tester",
            email=email,
            password=password,
            password_confirm=password
        ),
        db=db_session
    )
    assert reg_res.success is True
    assert reg_res.data.account_status == AccountStatus.PENDING_VERIFICATION

    # 2. Extract Verification Token & Verify Email
    email_entry = notification_service.get_latest_email_for(email)
    assert email_entry is not None
    token = email_entry["text_body"].split("verification token is:")[-1].strip()

    ver_res = verify_email(VerifyEmailRequest(token=token), db=db_session)
    assert ver_res.success is True

    # 3. Login
    req_obj = MockRequest()
    login_res = login(
        LoginRequest(email=email, password=password),
        request=req_obj,
        db=db_session
    )
    assert login_res.success is True
    access_tok = login_res.data.access_token
    refresh_tok = login_res.data.refresh_token

    user = db_session.query(User).filter(User.id == login_res.data.user_id).one()
    assert user.is_verified is True
    assert user.account_status == AccountStatus.ACTIVE

    # 4. View & Update Profile
    prof_res = update_current_user_profile(
        UpdateProfileRequest(bio="Passionate about humanitarian education.", phone_number="+1-555-0199"),
        current_user=user,
        db=db_session
    )
    assert prof_res.success is True
    assert prof_res.data.bio == "Passionate about humanitarian education."

    # 5. Organization Owner invites this user
    owner = User(email=f"org_owner_{ts}@ngo.org", full_name="Global NGO", hashed_password="hash", role=PlatformRole.DONOR, account_status=AccountStatus.ACTIVE, is_verified=True)
    org = Organization(name=f"E2E NGO {ts}", country="USA")
    db_session.add_all([owner, org])
    db_session.flush()

    mem_owner = OrganizationMembership(user_id=owner.id, organization_id=org.id, role=OrganizationRole.OWNER, is_active=True)
    db_session.add(mem_owner)
    db_session.commit()

    inv_res = invite_member(
        organization_id=org.id,
        req=CreateInvitationRequest(invitee_email=email, role=OrganizationRole.MEMBER),
        current_user=owner,
        db=db_session
    )
    assert inv_res.success is True

    # 6. User accepts invitation
    acc_res = accept_invitation_by_id(invitation_id=inv_res.data.id, current_user=user, db=db_session)
    assert acc_res.success is True
    assert acc_res.data.organization_id == org.id

    # Verify membership list
    my_orgs = get_my_organizations(current_user=user, db=db_session)
    assert len(my_orgs.data) == 1
    assert my_orgs.data[0].role == OrganizationRole.MEMBER

    # 7. Rotate Refresh Token
    rotate_res = refresh_token(RefreshTokenRequest(refresh_token=refresh_tok), request=req_obj, db=db_session)
    assert rotate_res.success is True
    new_refresh = rotate_res.data.refresh_token
    assert new_refresh != refresh_tok

    # 8. Change Password
    chg_res = change_password(
        ChangePasswordRequest(
            current_password=password,
            new_password="UpdatedPassword123!",
            new_password_confirm="UpdatedPassword123!"
        ),
        current_user=user,
        db=db_session
    )
    assert chg_res.success is True

    # 9. Logout
    req_with_auth = MockRequest(headers={"Authorization": f"Bearer {rotate_res.data.access_token}"})
    logout_res = logout(request=req_with_auth, current_user=user, db=db_session)
    assert logout_res.success is True

