"""
admin.py - Administrative Identity Management API Endpoints
============================================================
Routes for platform administrator inspection, user status management,
role elevation, and security audit review.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.core.constants import PlatformRole, AccountStatus
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import require_platform_role
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.admin_service import AdminService
from backend.app.api.schemas import (
    StandardResponse, PaginatedResponse, PaginationMeta,
    UserResponse, UpdateUserStatusRequest, UpdateUserRoleRequest
)

router = APIRouter(
    prefix="/admin",
    tags=["Administrative Identity Management"],
    dependencies=[Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN))]
)


@router.get("/users", response_model=PaginatedResponse[UserResponse])
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[AccountStatus] = None,
    role: Optional[PlatformRole] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """Lists platform users with status, role, and search filters."""
    users, total = AdminService.list_users(
        db, page=page, page_size=page_size, status=status, role=role, search=search
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return PaginatedResponse(
        data=users,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=total_pages
        ),
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/users/{user_id}", response_model=StandardResponse[UserResponse])
def get_user(user_id: str, db: Session = Depends(get_db_session)):
    """Retrieves full details for a specified user."""
    data = AdminService.get_user_details(db, user_id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/users/{user_id}/status", response_model=StandardResponse[dict])
def update_user_status(
    user_id: str,
    req: UpdateUserStatusRequest,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Updates user status (e.g. SUSPENDED). Revokes active sessions if suspended."""
    AdminService.update_user_status(db, user_id, req.status, current_admin.id)
    return StandardResponse(
        data={"updated": True},
        message=f"User status updated to {req.status.value}.",
        correlation_id=correlation_id_ctx.get()
    )


@router.patch("/users/{user_id}/role", response_model=StandardResponse[dict])
def update_user_role(
    user_id: str,
    req: UpdateUserRoleRequest,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Assigns platform role (DONOR, REVIEWER, PLATFORM_ADMIN) to a user."""
    AdminService.update_user_role(db, user_id, req.role, current_admin.id)
    return StandardResponse(
        data={"updated": True},
        message=f"User role updated to {req.role.value}.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/security-events", response_model=StandardResponse[List[dict]])
def list_security_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    action: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """Inspects platform audit events for security and compliance reviews."""
    events, total = AdminService.list_security_events(db, page=page, page_size=page_size, action=action)
    return StandardResponse(data=events, correlation_id=correlation_id_ctx.get())

