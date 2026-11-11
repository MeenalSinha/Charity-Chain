"""
test_admin_api.py - Integration tests for Platform Administrative Identity Operations
=====================================================================================
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.models.users import User, AccountStatus
from backend.app.core.constants import PlatformRole
from backend.app.api.schemas import UpdateUserStatusRequest, UpdateUserRoleRequest
from backend.app.api.v1.admin import (
    list_users, get_user, update_user_status, update_user_role, list_security_events
)


@pytest.fixture
def admin_user(db_session):
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    admin = User(
        email=f"admin_{ts}@charitychain.org",
        full_name="Super Admin",
        hashed_password="hash",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(admin)
    db_session.commit()
    return admin


def test_admin_user_lifecycle_controls(admin_user, db_session):
    """Verify admin listing, status update to SUSPENDED, and role update."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    target_user = User(
        email=f"target_{ts}@charitychain.org",
        full_name="Target User",
        hashed_password="hash",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(target_user)
    db_session.commit()

    # 1. Admin list users
    user_list = list_users(page=1, page_size=10, search="Target", db=db_session)
    assert user_list.success is True
    assert any(u.id == target_user.id for u in user_list.data)

    # 2. Admin get user details
    details = get_user(user_id=target_user.id, db=db_session)
    assert details.success is True
    assert details.data.email == target_user.email

    # 3. Admin suspends user
    susp_res = update_user_status(
        user_id=target_user.id,
        req=UpdateUserStatusRequest(status=AccountStatus.SUSPENDED),
        current_admin=admin_user,
        db=db_session
    )
    assert susp_res.success is True
    assert target_user.account_status == AccountStatus.SUSPENDED

    # 4. Admin updates user role to REVIEWER
    role_res = update_user_role(
        user_id=target_user.id,
        req=UpdateUserRoleRequest(role=PlatformRole.REVIEWER),
        current_admin=admin_user,
        db=db_session
    )
    assert role_res.success is True
    assert target_user.role == PlatformRole.REVIEWER

    # 5. Admin lists security events
    events_res = list_security_events(page=1, page_size=20, db=db_session)
    assert events_res.success is True
    assert len(events_res.data) > 0

