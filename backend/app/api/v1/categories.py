"""
categories.py - Project Categories REST API Endpoints
=====================================================
Public category discovery and administrative category creation.
"""

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.core.constants import PlatformRole
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user, require_platform_role
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.project_service import ProjectService
from backend.app.api.schemas import (
    StandardResponse, ProjectCategoryResponse,
    CreateProjectCategoryRequest
)

router = APIRouter(tags=["Project Categories"])


@router.get("/project-categories", response_model=StandardResponse[List[ProjectCategoryResponse]])
def list_categories(
    db: Session = Depends(get_db_session)
):
    """Lists all active project categories for project creation and filtering."""
    data = ProjectService.list_categories(db, include_inactive=False)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post(
    "/admin/project-categories",
    response_model=StandardResponse[ProjectCategoryResponse],
    status_code=status.HTTP_201_CREATED
)
def create_category(
    req: CreateProjectCategoryRequest,
    current_admin: User = Depends(require_platform_role(PlatformRole.PLATFORM_ADMIN)),
    db: Session = Depends(get_db_session)
):
    """Creates a new database-backed project category. Requires Platform Admin."""
    data = ProjectService.create_category(db, current_admin.id, req)
    return StandardResponse(
        data=data,
        message=f"Category '{data.name}' created.",
        correlation_id=correlation_id_ctx.get()
    )

