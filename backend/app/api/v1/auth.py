"""
auth.py - Authentication & Session API Endpoints
================================================
Routes for registration, verification, login, refresh rotation, logout, and password recovery.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.app.core.config import settings
from backend.app.core.logging import correlation_id_ctx
from backend.app.core.permissions import get_current_user, get_current_active_user
from backend.app.db.session import get_db_session
from backend.app.db.models.users import User, UserSession
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import (
    StandardResponse, RegisterRequest, RegisterResponse,
    VerifyEmailRequest, ResendVerificationRequest, LoginRequest,
    TokenResponse, RefreshTokenRequest, ForgotPasswordRequest,
    ResetPasswordRequest, ChangePasswordRequest, SessionResponse,
    UserResponse, UserProfileResponse
)

router = APIRouter(prefix="/auth", tags=["Authentication & Identity"])


@router.post("/register", response_model=StandardResponse[RegisterResponse], status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db_session)):
    """Registers a new user account with default DONOR role."""
    data = AuthService.register_user(db, req)
    return StandardResponse(
        data=data,
        message="Registration successful. Please verify your email.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/verify-email", response_model=StandardResponse[dict])
def verify_email(req: VerifyEmailRequest, db: Session = Depends(get_db_session)):
    """Verifies user email with cryptographic single-use token."""
    AuthService.verify_email(db, req.token)
    return StandardResponse(
        data={"verified": True},
        message="Email verification successful. Account is now active.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/resend-verification", response_model=StandardResponse[dict])
def resend_verification(req: ResendVerificationRequest, db: Session = Depends(get_db_session)):
    """Dispatches a fresh email verification link."""
    AuthService.resend_verification(db, req.email)
    return StandardResponse(
        data={"sent": True},
        message="If an unverified account exists for this email, a verification link has been sent.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/login", response_model=StandardResponse[TokenResponse])
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db_session)):
    """Authenticates credentials, manages failed attempt lockouts, and issues tokens."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = AuthService.authenticate_user(db, req, ip_address=client_ip, user_agent=user_agent)
    return StandardResponse(
        data=tokens,
        message="Authentication successful.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/refresh", response_model=StandardResponse[TokenResponse])
def refresh_token(req: RefreshTokenRequest, request: Request, db: Session = Depends(get_db_session)):
    """Rotates refresh token and issues fresh access token."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = AuthService.rotate_refresh_token(db, req.refresh_token, ip_address=client_ip, user_agent=user_agent)
    return StandardResponse(
        data=tokens,
        message="Token refreshed successfully.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/logout", response_model=StandardResponse[dict])
def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Revokes the current session."""
    # If session was validated via token payload
    token_str = request.headers.get("Authorization", "").replace("Bearer ", "")
    from backend.app.core.security import decode_access_token
    session_id = None
    try:
        p = decode_access_token(token_str)
        session_id = p.get("session_id")
    except Exception:
        pass

    if session_id:
        AuthService.logout(db, session_id, current_user.id)
    return StandardResponse(
        data={"logged_out": True},
        message="Logged out successfully.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/logout-all", response_model=StandardResponse[dict])
def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """Revokes all active sessions for the authenticated user."""
    AuthService.logout_all_sessions(db, current_user.id)
    return StandardResponse(
        data={"logged_out_all": True},
        message="All active sessions have been revoked.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/sessions", response_model=StandardResponse[List[SessionResponse]])
def list_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Lists all active and historical sessions for the current user."""
    sessions = db.execute(
        select(UserSession)
        .where(UserSession.user_id == current_user.id)
        .order_by(UserSession.last_used_at.desc())
    ).scalars().all()

    data = [
        SessionResponse(
            id=s.id,
            ip_address=s.ip_address,
            user_agent=s.user_agent,
            last_used_at=s.last_used_at,
            expires_at=s.expires_at,
            is_revoked=s.is_revoked
        )
        for s in sessions
    ]
    return StandardResponse(data=data, correlation_id=correlation_id_ctx.get())


@router.post("/sessions/{session_id}/revoke", response_model=StandardResponse[dict])
def revoke_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Revokes a specific session belonging to the current user."""
    AuthService.logout(db, session_id, current_user.id)
    return StandardResponse(
        data={"revoked": True},
        message="Session revoked.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/forgot-password", response_model=StandardResponse[dict])
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db_session)):
    """Dispatches a password reset recovery link."""
    AuthService.initiate_password_reset(db, req.email)
    return StandardResponse(
        data={"requested": True},
        message="If an account exists for this email, password recovery instructions have been sent.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/reset-password", response_model=StandardResponse[dict])
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db_session)):
    """Resets password using single-use recovery token."""
    AuthService.reset_password(db, req.token, req.new_password, req.new_password_confirm)
    return StandardResponse(
        data={"reset": True},
        message="Password has been reset successfully. Please log in with your new password.",
        correlation_id=correlation_id_ctx.get()
    )


@router.post("/change-password", response_model=StandardResponse[dict])
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Changes password for authenticated user and invalidates other sessions."""
    AuthService.change_password(db, current_user.id, req.current_password, req.new_password, req.new_password_confirm)
    return StandardResponse(
        data={"changed": True},
        message="Password changed successfully.",
        correlation_id=correlation_id_ctx.get()
    )


@router.get("/me", response_model=StandardResponse[UserResponse])
def get_me(current_user: User = Depends(get_current_user)):
    """Returns current authenticated user details."""
    prof_dto = None
    if current_user.profile:
        prof_dto = UserProfileResponse(
            bio=current_user.profile.bio,
            phone_number=current_user.profile.phone_number,
            avatar_url=current_user.profile.avatar_url
        )
    user_dto = UserResponse(
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
    return StandardResponse(data=user_dto, correlation_id=correlation_id_ctx.get())

