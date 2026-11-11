"""
projects.py - Charitable Projects REST API Endpoints
====================================================
Public discovery, lifecycle management, structured validation, and project updates.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user, get_current_user_optional
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.project_service import ProjectService
from backend.app.api.schemas import (
    StandardResponse, PaginatedResponse, ProjectResponse,
    UpdateProjectRequest, ProjectValidationResponse
)

router = APIRouter(prefix="/projects", tags=["Charitable Projects & Discovery"])


def _clean_param(val, default=None):
    if val is None:
        return default
    if hasattr(val, "default"):
        return val.default if val.default is not ... else default
    return val


@router.get("", response_model=PaginatedResponse[ProjectResponse])
def discover_projects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: Optional[str] = Query(default=None),
    category_id: Optional[str] = Query(default=None),
    country: Optional[str] = Query(default=None),
    min_target: Optional[int] = Query(default=None, ge=0),
    max_target: Optional[int] = Query(default=None, ge=0),
    db: Session = Depends(get_db_session)
):
    """
    Public project discovery endpoint.
    Returns only ACTIVE, public projects belonging to VERIFIED organizations.
    Supports full-text search, category, country, and funding target filtering.
    """
    res = ProjectService.discover_projects(
        db=db,
        page=_clean_param(page, 1),
        page_size=_clean_param(page_size, 20),
        search=_clean_param(search, None),
        category_id=_clean_param(category_id, None),
        country=_clean_param(country, None),
        min_target=_clean_param(min_target, None),
        max_target=_clean_param(max_target, None)
    )
    res.correlation_id = correlation_id_ctx.get()
    return res


@router.get("/{project_id}", response_model=StandardResponse[ProjectResponse])
def get_project_details(
    project_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db_session)
):
    """
    Retrieves project details.
    Publicly visible if project is ACTIVE and PUBLIC;
    Draft, under-review, or private projects require membership or reviewer authorization.
    """
    data = ProjectService.get_project(db, project_id, current_user.id if current_user else None)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/{project_id}", response_model=StandardResponse[ProjectResponse])
def update_project(
    project_id: str,
    req: UpdateProjectRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates project details. Permitted when project is in DRAFT or REVISION_REQUIRED."""
    data = ProjectService.update_project(db, project_id, current_user.id, req)
    return StandardResponse(data=data, message="Project updated successfully.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/validate", response_model=StandardResponse[ProjectValidationResponse])
def validate_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Runs comprehensive pre-submission validation.
    Returns structured report of all validation issues and whether project can be submitted.
    """
    data = ProjectService.validate_project_submission(db, project_id)
    msg = "Project satisfies all submission criteria." if data.valid else f"Project has {len(data.issues)} submission issue(s)."
    return StandardResponse(data=data, message=msg, correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/submit", response_model=StandardResponse[ProjectResponse])
def submit_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Submits project for reviewer approval after validating requirements."""
    data = ProjectService.submit_project(db, project_id, current_user.id)
    return StandardResponse(
        data=data,
        message=f"Project '{data.title}' successfully submitted for review.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/{project_id}/withdraw", response_model=StandardResponse[ProjectResponse])
def withdraw_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Withdraws a submitted project back to DRAFT for further editing."""
    data = ProjectService.withdraw_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message="Project withdrawn to DRAFT status.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/activate", response_model=StandardResponse[ProjectResponse])
def activate_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Activates an approved project, publishing it to public discovery."""
    data = ProjectService.activate_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message=f"Project '{data.title}' is now ACTIVE.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/pause", response_model=StandardResponse[ProjectResponse])
def pause_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Pauses an active project."""
    data = ProjectService.pause_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message="Project paused.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/resume", response_model=StandardResponse[ProjectResponse])
def resume_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Resumes a paused project."""
    data = ProjectService.resume_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message="Project resumed to ACTIVE.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/complete", response_model=StandardResponse[ProjectResponse])
def complete_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Marks project as completed."""
    data = ProjectService.complete_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message="Project marked as COMPLETED.", correlation_id=correlation_id_ctx.get())


@router.post("/{project_id}/archive", response_model=StandardResponse[ProjectResponse])
def archive_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Archives project and removes it from active operations."""
    data = ProjectService.archive_project(db, project_id, current_user.id)
    return StandardResponse(data=data, message="Project archived.", correlation_id=correlation_id_ctx.get())

