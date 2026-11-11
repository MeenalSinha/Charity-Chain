"""
test_auth_api.py - Integration tests for Authentication endpoints
=================================================================
Tests registration, duplicate email handling, verification, login,
refresh token rotation, logout, and password recovery.
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.session import SessionFactory
from backend.app.db.models.users import User, AccountStatus
from backend.app.core.constants import PlatformRole
from backend.app.core.exceptions import ConflictException, ValidationException, AuthenticationException
from backend.app.integrations.notifications.dev_sink import notification_service
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import (
    RegisterRequest, VerifyEmailRequest, LoginRequest,
    RefreshTokenRequest, ForgotPasswordRequest, ResetPasswordRequest,
    ChangePasswordRequest
)
from backend.app.api.v1.auth import (
    register, verify_email, login, refresh_token,
    logout, logout_all, forgot_password, reset_password, change_password, get_me
)


class MockRequest:
    def __init__(self, host="127.0.0.1", headers=None):
        class Client:
            def __init__(self, host):
                self.host = host
        self.client = Client(host)
        self.headers = headers or {}


def test_registration_and_email_verification_flow(db_session):
    """Verify registration sends verification email, and token verifies account."""
    notification_service.clear()
    email = f"user_{datetime.now(timezone.utc).timestamp()}@charitychain.org"
    reg_req = RegisterRequest(
        full_name="Alice Donor",
        email=email,
        password="ValidPassword123!",
        password_confirm="ValidPassword123!"
    )

    # 1. Register User
    res = register(reg_req, db=db_session)
    assert res.success is True
    assert res.data.email == email.lower()
    assert res.data.role == PlatformRole.DONOR
    assert res.data.account_status == AccountStatus.PENDING_VERIFICATION

    # 2. Extract verification token from dev sink email
    sent = notification_service.get_latest_email_for(email)
    assert sent is not None
    assert "verification token is:" in sent["text_body"]
    raw_token = sent["text_body"].split("verification token is:")[-1].strip()

    # 3. Verify Email
    ver_res = verify_email(VerifyEmailRequest(token=raw_token), db=db_session)
    assert ver_res.success is True
    assert ver_res.data["verified"] is True

    # 4. Check user is now ACTIVE in database
    user = db_session.query(User).filter(User.id == res.data.user_id).one()
    assert user.is_verified is True
    assert user.account_status == AccountStatus.ACTIVE


def test_duplicate_registration_rejected(db_session):
    """Verify duplicate email registration raises ConflictException."""
    email = f"dup_{datetime.now(timezone.utc).timestamp()}@charitychain.org"
    reg_req = RegisterRequest(
        full_name="Duplicate Tester",
        email=email,
        password="ValidPassword123!",
        password_confirm="ValidPassword123!"
    )
    register(reg_req, db=db_session)

    # Attempt registration with same email (different case)
    dup_req = RegisterRequest(
        full_name="Duplicate Tester 2",
        email=email.upper(),
        password="ValidPassword123!",
        password_confirm="ValidPassword123!"
    )
    with pytest.raises(ConflictException) as exc:
        register(dup_req, db=db_session)
    assert "already exists" in str(exc.value)


def test_login_and_refresh_flow(db_session):
    """Verify authenticated login, token return, and refresh rotation."""
    email = f"logintest_{datetime.now(timezone.utc).timestamp()}@charitychain.org"
    # Register and manually activate
    reg_req = RegisterRequest(
        full_name="Login Tester",
        email=email,
        password="ValidPassword123!",
        password_confirm="ValidPassword123!"
    )
    res = register(reg_req, db=db_session)
    user = db_session.query(User).filter(User.id == res.data.user_id).one()
    user.is_verified = True
    user.account_status = AccountStatus.ACTIVE
    db_session.commit()

    # 1. Login
    req_obj = MockRequest()
    login_resp = login(LoginRequest(email=email, password="ValidPassword123!"), request=req_obj, db=db_session)
    assert login_resp.success is True
    assert login_resp.data.access_token is not None
    assert login_resp.data.refresh_token is not None

    # 2. Refresh Token
    ref_resp = refresh_token(RefreshTokenRequest(refresh_token=login_resp.data.refresh_token), request=req_obj, db=db_session)
    assert ref_resp.success is True
    assert ref_resp.data.access_token is not None
    assert ref_resp.data.refresh_token != login_resp.data.refresh_token

    # 3. Get Me
    me_resp = get_me(current_user=user)
    assert me_resp.success is True
    assert me_resp.data.email == email.lower()


def test_password_reset_flow(db_session):
    """Verify forgot password initiates reset token and reset updates password."""
    notification_service.clear()
    email = f"resettest_{datetime.now(timezone.utc).timestamp()}@charitychain.org"
    reg_req = RegisterRequest(
        full_name="Reset Tester",
        email=email,
        password="OldPassword123!",
        password_confirm="OldPassword123!"
    )
    res = register(reg_req, db=db_session)
    user = db_session.query(User).filter(User.id == res.data.user_id).one()
    user.is_verified = True
    user.account_status = AccountStatus.ACTIVE
    db_session.commit()

    # 1. Forgot Password
    forgot_password(ForgotPasswordRequest(email=email), db=db_session)
    sent = notification_service.get_latest_email_for(email)
    assert sent is not None
    assert "token to reset your password" in sent["html_body"]
    import re
    token_match = re.search(r"<b>(.*?)</b>", sent["html_body"])
    reset_tok = token_match.group(1)

    # 2. Reset Password
    reset_password(
        ResetPasswordRequest(
            token=reset_tok,
            new_password="BrandNewPassword123!",
            new_password_confirm="BrandNewPassword123!"
        ),
        db=db_session
    )

    # 3. Verify login works with new password
    req_obj = MockRequest()
    login_resp = login(LoginRequest(email=email, password="BrandNewPassword123!"), request=req_obj, db=db_session)
    assert login_resp.success is True

