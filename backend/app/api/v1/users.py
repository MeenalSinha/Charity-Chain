"""
users.py - User Profile Management API Endpoints
================================================
Endpoints for viewing and updating current user account details and profile attributes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User, UserProfile
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.api.schemas import (
    StandardResponse, UserResponse, UserProfileResponse,
    UpdateUserRequest, UpdateProfileRequest
)

router = APIRouter(prefix="/users", tags=["Users & Profiles"])


@router.get("/me", response_model=StandardResponse[UserResponse])
def get_current_user_account(current_user: User = Depends(get_current_active_user)):
    """Returns the authenticated user's account details."""
    prof_dto = None
    if current_user.profile:
        prof_dto = UserProfileResponse(
            bio=current_user.profile.bio,
            phone_number=current_user.profile.phone_number,
            avatar_url=current_user.profile.avatar_url
        )
    data = UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        account_status=current_user.account_status,
        is_verified=current_user.is_verified,
        email_verified_at=current_user.email_verified_at,
        created_at=current_user.created_at,
        profile=prof_dto
    )
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/me", response_model=StandardResponse[UserResponse])
def update_current_user_account(
    req: UpdateUserRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates permitted account fields (e.g., full_name). Platform role & status cannot be modified here."""
    if req.full_name is not None:
        current_user.full_name = req.full_name.strip()
        audit = AuditEvent(
            entity_name="User",
            entity_id=current_user.id,
            action="USER_NAME_UPDATED",
            actor_id=current_user.id,
            after_state={"full_name": current_user.full_name}
        )
        db.add(audit)
        db.commit()

    prof_dto = None
    if current_user.profile:
        prof_dto = UserProfileResponse(
            bio=current_user.profile.bio,
            phone_number=current_user.profile.phone_number,
            avatar_url=current_user.profile.avatar_url
        )

    data = UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        account_status=current_user.account_status,
        is_verified=current_user.is_verified,
        email_verified_at=current_user.email_verified_at,
        created_at=current_user.created_at,
        profile=prof_dto
    )
    return StandardResponse(data=data, message="Account updated successfully.", correlation_id=correlation_id_ctx.get())


@router.get("/me/profile", response_model=StandardResponse[UserProfileResponse])
def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Returns profile metadata (bio, phone number, avatar URL)."""
    if not current_user.profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        db.commit()
    else:
        profile = current_user.profile

    data = UserProfileResponse(
        bio=profile.bio,
        phone_number=profile.phone_number,
        avatar_url=profile.avatar_url
    )
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.patch("/me/profile", response_model=StandardResponse[UserProfileResponse])
def update_current_user_profile(
    req: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Updates profile attributes."""
    if not current_user.profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
    else:
        profile = current_user.profile

    if req.bio is not None:
        profile.bio = req.bio
    if req.phone_number is not None:
        profile.phone_number = req.phone_number
    if req.avatar_url is not None:
        profile.avatar_url = req.avatar_url

    db.commit()
    data = UserProfileResponse(
        bio=profile.bio,
        phone_number=profile.phone_number,
        avatar_url=profile.avatar_url
    )
    return StandardResponse(data=data, message="Profile updated successfully.", correlation_id=correlation_id_ctx.get())

