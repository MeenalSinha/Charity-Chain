"""
milestones.py - Multi-Milestone Management REST API Endpoints
=============================================================
Routes for adding, listing, updating, and removing project milestones.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User
from backend.app.services.project_service import ProjectService
from backend.app.api.schemas import (
    StandardResponse, CreateMilestoneRequest,
    UpdateMilestoneRequest, MilestoneResponse
)

router = APIRouter(tags=["Project Milestones"])


@router.get("/milestones", response_model=StandardResponse[List[MilestoneResponse]])
def list_all_milestones(
    project_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db_session)
):
    """Lists milestones, optionally filtered by project_id."""
    clean_id = project_id if (project_id is not None and not hasattr(project_id, "default")) else None
    if clean_id:
        data = ProjectService.list_milestones(db, clean_id)
    else:
        from backend.app.db.models.projects import Milestone
        from sqlalchemy import select
        milestones = db.execute(select(Milestone).limit(100)).scalars().all()
        data = [ProjectService._build_milestone_response(m) for m in milestones]
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post("/projects/{project_id}/milestones", response_model=StandardResponse[MilestoneResponse], status_code=status.HTTP_201_CREATED)
def create_milestone(
    project_id: str,
    req: CreateMilestoneRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Adds a new milestone to a project. Enforces sequence and budget validation."""
    data = ProjectService.create_milestone(db, project_id, current_user.id, req)
    return StandardResponse(
        data=data,
        message=f"Milestone '{data.title}' (#{data.sequence_number}) added to project.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/projects/{project_id}/milestones", response_model=StandardResponse[List[MilestoneResponse]])
def list_project_milestones(
    project_id: str,
    db: Session = Depends(get_db_session)
):
    """Lists all milestones belonging to a project ordered by sequence number."""
    data = ProjectService.list_milestones(db, project_id)
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/milestones/{milestone_id}", response_model=StandardResponse[MilestoneResponse])
def update_milestone(
    milestone_id: str,
    req: UpdateMilestoneRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates an existing milestone prior to project submission."""
    data = ProjectService.update_milestone(db, milestone_id, current_user.id, req)
    return StandardResponse(data=data, message="Milestone updated successfully.", correlation_id=correlation_id_ctx.get())


@router.delete("/milestones/{milestone_id}", response_model=StandardResponse[dict])
def delete_milestone(
    milestone_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Deletes a milestone and automatically re-indexes remaining sequence numbers."""
    ProjectService.delete_milestone(db, milestone_id, current_user.id)
    return StandardResponse(data={"deleted": True}, message="Milestone deleted.", correlation_id=correlation_id_ctx.get())
