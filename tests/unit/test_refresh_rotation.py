"""
test_refresh_rotation.py - Unit tests for refresh token rotation & reuse detection
"""

import pytest
from sqlalchemy import select
from backend.app.db.session import SessionFactory
from backend.app.db.models.users import User, UserSession
from backend.app.core.constants import PlatformRole, AccountStatus
from backend.app.core.security import hash_password, AuthenticationException
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import LoginRequest


@pytest.fixture
def active_user(db_session):
    """Creates a verified active user for session testing."""
    email = "session_test_user@charitychain.org"
    user = db_session.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if not user:
        user = User(
            email=email,
            full_name="Session Tester",
            hashed_password=hash_password("ValidPassword123!"),
            role=PlatformRole.DONOR,
            account_status=AccountStatus.ACTIVE,
            is_verified=True
        )
        db_session.add(user)
        db_session.commit()
    return user


def test_login_creates_session(active_user, db_session):
    """Verify login creates persistent UserSession with token family."""
    login_resp = AuthService.authenticate_user(
        db_session,
        LoginRequest(email=active_user.email, password="ValidPassword123!"),
        ip_address="127.0.0.1",
        user_agent="PyTestClient"
    )
    assert login_resp.access_token is not None
    assert login_resp.refresh_token is not None

    # Check session record
    session = db_session.execute(
        select(UserSession).where(UserSession.user_id == active_user.id)
    ).scalars().all()
    assert len(session) >= 1
    latest = session[-1]
    assert latest.is_revoked is False
    assert latest.token_family is not None


def test_refresh_token_rotation_and_reuse_detection(active_user, db_session):
    """Verify refresh token rotation and family revocation on reuse attempt."""
    # 1. Login to get initial refresh token
    login_resp = AuthService.authenticate_user(
        db_session,
        LoginRequest(email=active_user.email, password="ValidPassword123!")
    )
    r1 = login_resp.refresh_token

    # 2. Perform legitimate rotation
    rotate_resp = AuthService.rotate_refresh_token(db_session, r1)
    r2 = rotate_resp.refresh_token
    assert r2 != r1

    # 3. Old token r1 should now be revoked
    from backend.app.core.security import hash_token
    h1 = hash_token(r1)
    s1 = db_session.execute(
        select(UserSession).where(UserSession.refresh_token_hash == h1)
    ).scalar_one()
    assert s1.is_revoked is True
    assert s1.revocation_reason == "ROTATED"

    # 4. REUSE DETECTION: Presenting old token r1 again must trigger family compromise revocation!
    with pytest.raises(AuthenticationException) as exc_info:
        AuthService.rotate_refresh_token(db_session, r1)
    
    assert "Compromised refresh token" in str(exc_info.value)

    # 5. Verify that r2's session in the same family was also revoked due to reuse detection!
    h2 = hash_token(r2)
    s2 = db_session.execute(
        select(UserSession).where(UserSession.refresh_token_hash == h2)
    ).scalar_one()
    assert s2.is_revoked is True
    assert s2.revocation_reason == "TOKEN_REUSE_DETECTED"

