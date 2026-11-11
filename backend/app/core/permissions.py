"""
permissions.py - Centralized Authorization & RBAC Engine
=========================================================
Implements user resolution, session liveness verification, platform-role checks,
organization-scoped access control, and resource ownership protection.
"""

from typing import List, Optional, Callable
from datetime import datetime

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.constants import PlatformRole, OrganizationRole, AccountStatus
from backend.app.core.exceptions import AuthenticationException, ForbiddenException
from backend.app.core.security import decode_access_token
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User, UserSession, OrganizationMembership

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db_session)
) -> User:
    """
    Resolves the current authenticated user from JWT token and verifies session validity.
    """
    if not token:
        raise AuthenticationException(
            "Authentication credentials were not provided.",
            code="AUTHENTICATION_REQUIRED",
            status_code=401
        )

    payload = decode_access_token(token)
    user_id = payload.get("sub")
    session_id = payload.get("session_id")

    if not user_id:
        raise AuthenticationException("Invalid token payload: missing subject.", code="INVALID_TOKEN", status_code=401)

    # Verify session liveness if session_id is encoded
    if session_id:
        sess = db.execute(
            select(UserSession).where(UserSession.id == session_id)
        ).scalar_one_or_none()

        if not sess or sess.is_revoked:
            raise AuthenticationException(
                "Authentication session has been revoked or expired.",
                code="SESSION_REVOKED",
                status_code=401
            )
        if sess.expires_at < datetime.utcnow():
            raise AuthenticationException(
                "Authentication session has expired.",
                code="SESSION_EXPIRED",
                status_code=401
            )

    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not user:
        raise AuthenticationException("User account not found.", code="USER_NOT_FOUND", status_code=401)

    return user


def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    """Ensures the authenticated user has verified email and an active account status."""
    if user.account_status == AccountStatus.PENDING_VERIFICATION or not user.is_verified:
        raise ForbiddenException("Email address is not verified. Please verify your email first.")

    if user.account_status in (AccountStatus.SUSPENDED, AccountStatus.DEACTIVATED):
        raise ForbiddenException("Your account has been suspended or deactivated.")

    if user.account_status == AccountStatus.LOCKED or (user.locked_until and user.locked_until > datetime.utcnow()):
        raise ForbiddenException("Your account is temporarily locked.")

    return user


def require_platform_role(*allowed_roles: PlatformRole) -> Callable:
    """
    FastAPI dependency factory enforcing platform-level roles.
    PLATFORM_ADMIN automatically bypasses platform role restrictions.
    """
    def role_checker(user: User = Depends(get_current_active_user)) -> User:
        if user.role == PlatformRole.PLATFORM_ADMIN:
            return user
        if user.role not in allowed_roles:
            raise ForbiddenException(
                f"Action forbidden. Required platform role: {[r.value for r in allowed_roles]}.",
                details={"user_role": user.role.value}
            )
        return user
    return role_checker


def require_org_role(
    org_id: str,
    user_id: str,
    db: Session,
    allowed_roles: List[OrganizationRole]
) -> OrganizationMembership:
    """
    Verifies that a user has one of the required roles in the given organization.
    Platform administrators bypass organization membership checks.
    """
    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if user and user.role == PlatformRole.PLATFORM_ADMIN:
        # Return a synthetic or owner-level authorization context for platform admins
        mem = db.execute(
            select(OrganizationMembership).where(
                and_(OrganizationMembership.organization_id == org_id, OrganizationMembership.user_id == user_id)
            )
        ).scalar_one_or_none()
        if mem:
            return mem
        return OrganizationMembership(user_id=user_id, organization_id=org_id, role=OrganizationRole.OWNER)

    membership = db.execute(
        select(OrganizationMembership).where(
            and_(
                OrganizationMembership.organization_id == org_id,
                OrganizationMembership.user_id == user_id,
                OrganizationMembership.is_active == True
            )
        )
    ).scalar_one_or_none()

    if not membership:
        raise ForbiddenException("You are not a member of this organization.")

    if membership.role not in allowed_roles:
        raise ForbiddenException(
            f"Action forbidden. Required organization role: {[r.value for r in allowed_roles]}.",
            details={"current_role": membership.role.value}
        )

    return membership


def verify_resource_ownership(resource_owner_id: str, current_user: User) -> None:
    """
    Verifies that the current user owns the specified resource or is a platform admin.
    Protects against Insecure Direct Object References (IDOR).
    """
    if current_user.role == PlatformRole.PLATFORM_ADMIN:
        return
    if current_user.id != resource_owner_id:
        raise ForbiddenException("You do not have permission to access or modify this resource.")

