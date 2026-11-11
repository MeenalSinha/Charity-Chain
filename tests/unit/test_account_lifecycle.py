"""
test_account_lifecycle.py - Unit tests for account lifecycle status & lockout
"""

import pytest
from datetime import datetime, timedelta
from backend.app.core.constants import AccountStatus, PlatformRole
from backend.app.core.exceptions import ForbiddenException
from backend.app.core.permissions import get_current_active_user
from backend.app.db.models.users import User


def test_active_verified_user_passes():
    """Verify active verified user passes liveness checks."""
    user = User(
        email="active@charity.org",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    assert get_current_active_user(user) == user


def test_unverified_email_user_blocked():
    """Verify user in PENDING_VERIFICATION or with is_verified=False is blocked."""
    unverified_user = User(
        email="unverified@charity.org",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.PENDING_VERIFICATION,
        is_verified=False
    )
    with pytest.raises(ForbiddenException) as exc:
        get_current_active_user(unverified_user)
    assert "verify your email" in str(exc.value)


def test_suspended_or_deactivated_user_blocked():
    """Verify suspended or deactivated user is blocked."""
    suspended_user = User(
        email="suspended@charity.org",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.SUSPENDED,
        is_verified=True
    )
    with pytest.raises(ForbiddenException) as exc:
        get_current_active_user(suspended_user)
    assert "suspended or deactivated" in str(exc.value)


def test_locked_user_blocked():
    """Verify user with locked_until timestamp in future is blocked."""
    locked_user = User(
        email="locked@charity.org",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True,
        locked_until=datetime.utcnow() + timedelta(minutes=10)
    )
    with pytest.raises(ForbiddenException) as exc:
        get_current_active_user(locked_user)
    assert "temporarily locked" in str(exc.value)

