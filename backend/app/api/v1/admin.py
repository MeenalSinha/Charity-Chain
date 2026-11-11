"""
admin.py - Administrative Identity Management API Endpoints
============================================================
Routes for platform administrator inspection, user status management,
role elevation, and security audit review.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.core.constants import PlatformRole, AccountStatus, OrganizationStatus
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import require_platform_role
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.admin_service import AdminService
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.api.schemas import (
    StandardResponse, PaginatedResponse, PaginationMeta,
    UserResponse, UpdateUserStatusRequest, UpdateUserRoleRequest,
    OrganizationDetailResponse, OrganizationDecisionRequest,
    OrganizationSuspensionRequest, ProjectResponse, ProjectDecisionRequest
)

router = APIRouter(
    prefix="/admin",
    tags=["Administrative & Reviewer Workflows"]
)


@router.get("/users", response_model=PaginatedResponse[UserResponse])
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[AccountStatus] = None,
    role: Optional[PlatformRole] = None,
    search: Optional[str] = None,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Lists platform users with status, role, and search filters. Requires Platform Admin."""
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
def get_user(
    user_id: str,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Retrieves full details for a specified user. Requires Platform Admin."""
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
    current_admin: Optional[User] = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Lists audit events for platform administrators."""
    clean_page = page if not hasattr(page, "default") else 1
    clean_size = page_size if not hasattr(page_size, "default") else 50
    clean_action = action if not hasattr(action, "default") else None
    events, total = AdminService.list_security_events(db, page=clean_page, page_size=clean_size, action=clean_action)
    return StandardResponse(data=events, correlation_id=correlation_id_ctx.get())


# --- Reviewer & Verification Queue Endpoints ---

@router.get("/organizations/review-queue", response_model=PaginatedResponse[OrganizationDetailResponse])
def list_organization_review_queue(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[OrganizationStatus] = None,
    current_reviewer: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Lists organization applications awaiting review. Requires Reviewer or Platform Admin."""
    res = NGOOnboardingService.list_review_queue(db, page=page, page_size=page_size, status_filter=status_filter)
    res.correlation_id = correlation_id_ctx.get()
    return res


@router.post("/organizations/{organization_id}/decision", response_model=StandardResponse[OrganizationDetailResponse])
def review_organization_decision(
    organization_id: str,
    req: OrganizationDecisionRequest,
    current_reviewer: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """
    Submits a reviewer decision (APPROVE, REJECT, REQUEST_REVISION) on an organization.
    Enforces conflict-of-interest prevention against self-approval.
    """
    data = NGOOnboardingService.review_organization(db, organization_id, current_reviewer.id, req)
    return StandardResponse(
        data=data,
        message=f"Organization decision recorded: {req.decision}.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/organizations/{organization_id}/suspend", response_model=StandardResponse[OrganizationDetailResponse])
def suspend_organization(
    organization_id: str,
    req: OrganizationSuspensionRequest,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Suspends an organization and restricts its projects. Requires Platform Admin."""
    data = NGOOnboardingService.suspend_organization(db, organization_id, current_admin.id, req.reason)
    return StandardResponse(
        data=data,
        message=f"Organization '{data.name}' has been SUSPENDED.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/organizations/{organization_id}/reactivate", response_model=StandardResponse[OrganizationDetailResponse])
def reactivate_organization(
    organization_id: str,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Reactivates a suspended organization to VERIFIED status. Requires Platform Admin."""
    data = NGOOnboardingService.reactivate_organization(db, organization_id, current_admin.id)
    return StandardResponse(
        data=data,
        message=f"Organization '{data.name}' has been reactivated.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/projects/review-queue", response_model=PaginatedResponse[ProjectResponse])
def list_project_review_queue(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_reviewer: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Lists submitted charitable projects awaiting review. Requires Reviewer or Platform Admin."""
    res = ProjectService.list_review_queue(db, page=page, page_size=page_size)
    res.correlation_id = correlation_id_ctx.get()
    return res


@router.post("/projects/{project_id}/decision", response_model=StandardResponse[ProjectResponse])
def review_project_decision(
    project_id: str,
    req: ProjectDecisionRequest,
    current_reviewer: User = Depends(require_platform_role(PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """
    Submits a reviewer decision (APPROVE, REJECT, REQUEST_REVISION) on a project.
    Enforces conflict-of-interest prevention against self-approval.
    """
    data = ProjectService.review_project(db, project_id, current_reviewer.id, req)
    return StandardResponse(
        data=data,
        message=f"Project review decision recorded: {req.decision}.",
        correlation_id=correlation_id_ctx.get()
    )

