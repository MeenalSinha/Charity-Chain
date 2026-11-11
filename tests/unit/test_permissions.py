"""
test_permissions.py - Unit tests for centralized authorization & RBAC engine
"""

import pytest
from backend.app.core.constants import PlatformRole, OrganizationRole, AccountStatus
from backend.app.core.exceptions import ForbiddenException
from backend.app.core.permissions import (
    require_platform_role, require_org_role, verify_resource_ownership
)
from backend.app.db.models.users import User, Organization, OrganizationMembership


def test_platform_role_permission_checks():
    """Verify platform role gate enforcement."""
    admin_user = User(role=PlatformRole.PLATFORM_ADMIN, account_status=AccountStatus.ACTIVE, is_verified=True)
    reviewer_user = User(role=PlatformRole.REVIEWER, account_status=AccountStatus.ACTIVE, is_verified=True)
    donor_user = User(role=PlatformRole.DONOR, account_status=AccountStatus.ACTIVE, is_verified=True)

    # 1. Platform Admin Gate
    admin_checker = require_platform_role(PlatformRole.PLATFORM_ADMIN)
    assert admin_checker(admin_user) == admin_user
    with pytest.raises(ForbiddenException):
        admin_checker(reviewer_user)
    with pytest.raises(ForbiddenException):
        admin_checker(donor_user)

    # 2. Reviewer Gate (permits Reviewer & Admin)
    reviewer_checker = require_platform_role(PlatformRole.REVIEWER)
    assert reviewer_checker(reviewer_user) == reviewer_user
    assert reviewer_checker(admin_user) == admin_user  # Admin bypasses
    with pytest.raises(ForbiddenException):
        reviewer_checker(donor_user)

    # 3. Donor Gate
    donor_checker = require_platform_role(PlatformRole.DONOR)
    assert donor_checker(donor_user) == donor_user
    assert donor_checker(admin_user) == admin_user


def test_resource_ownership_idor_protection():
    """Verify IDOR resource ownership protection."""
    user_a = User(id="user-a", role=PlatformRole.DONOR)
    user_b = User(id="user-b", role=PlatformRole.DONOR)
    admin = User(id="user-admin", role=PlatformRole.PLATFORM_ADMIN)

    # Owner accesses own resource -> Allowed
    verify_resource_ownership(resource_owner_id="user-a", current_user=user_a)

    # Admin accesses user resource -> Allowed
    verify_resource_ownership(resource_owner_id="user-a", current_user=admin)

    # Non-owner accesses user resource -> ForbiddenException
    with pytest.raises(ForbiddenException) as exc:
        verify_resource_ownership(resource_owner_id="user-a", current_user=user_b)
    assert "permission to access" in str(exc.value)

