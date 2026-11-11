"""
auth_service.py - Authentication & Identity Domain Service
==========================================================
Encapsulates registration, email verification, login with brute-force protection,
JWT generation, persistent session tracking, refresh token rotation, and password recovery.
"""

import uuid
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List

from sqlalchemy.orm import Session
from sqlalchemy import select, update, and_

from backend.app.core.config import settings
from backend.app.core.constants import PlatformRole, AccountStatus, TokenType
from backend.app.core.exceptions import (
    CharityChainException, ConflictException, NotFoundException,
    ValidationException, ForbiddenException
)
from backend.app.core.logging import logger
from backend.app.core.security import (
    hash_password, verify_password, validate_password_strength,
    normalize_email, generate_secure_token, hash_token, verify_token_hash,
    create_access_token, AuthenticationException, TokenExpiredException
)
from backend.app.db.models.users import (
    User, UserProfile, UserSession, EmailVerificationToken, PasswordResetToken
)
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.integrations.notifications.dev_sink import notification_service
from backend.app.api.schemas import (
    RegisterRequest, RegisterResponse, LoginRequest, TokenResponse
)

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$")


class AuthService:
    """Core Authentication & Identity Service."""

    @staticmethod
    def register_user(db: Session, req: RegisterRequest) -> RegisterResponse:
        """
        Registers a new user account with default DONOR role and PENDING_VERIFICATION status.
        Dispatches email verification token to notification sink.
        """
        # 1. Validate Email Syntax
        clean_email = normalize_email(req.email)
        if not clean_email or not EMAIL_REGEX.match(clean_email):
            raise ValidationException("Invalid email address syntax.", details={"email": req.email})

        # 2. Validate Password Match & Strength
        if req.password != req.password_confirm:
            raise ValidationException("Passwords do not match.", details={"field": "password_confirm"})

        valid_pwd, pwd_error = validate_password_strength(req.password)
        if not valid_pwd:
            raise ValidationException(pwd_error or "Password validation failed.", details={"field": "password"})

        # 3. Check Case-Insensitive Uniqueness
        existing = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        if existing:
            raise ConflictException("An account with this email address already exists.")

        # 4. Hash Password with Argon2id
        pwd_hash = hash_password(req.password)

        # 5. Create User & Profile Entities
        now = datetime.utcnow()
        new_user = User(
            email=clean_email,
            full_name=req.full_name.strip(),
            hashed_password=pwd_hash,
            role=PlatformRole.DONOR,
            account_status=AccountStatus.PENDING_VERIFICATION,
            is_verified=False,
            password_changed_at=now
        )
        db.add(new_user)
        db.flush()

        profile = UserProfile(user_id=new_user.id)
        db.add(profile)

        # 6. Generate Single-Use Email Verification Token
        raw_token = generate_secure_token(32)
        token_hash = hash_token(raw_token)
        verif_token = EmailVerificationToken(
            user_id=new_user.id,
            token_hash=token_hash,
            expires_at=now + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS)
        )
        db.add(verif_token)

        # 7. Audit Event
        audit = AuditEvent(
            entity_name="User",
            entity_id=new_user.id,
            action="USER_REGISTERED",
            actor_id=new_user.id,
            after_state={"email": clean_email, "role": PlatformRole.DONOR.value}
        )
        db.add(audit)
        db.commit()

        # 8. Send Verification Email
        notification_service.send_email_sync(
            recipient_email=clean_email,
            subject="Verify Your CharityChain Account",
            html_body=f"<p>Welcome to CharityChain. Your verification token is: <b>{raw_token}</b></p>",
            text_body=f"Welcome to CharityChain. Your verification token is: {raw_token}"
        )

        return RegisterResponse(
            user_id=new_user.id,
            email=new_user.email,
            full_name=new_user.full_name,
            role=new_user.role,
            account_status=new_user.account_status,
            requires_email_verification=True
        )

    @staticmethod
    def verify_email(db: Session, token: str) -> bool:
        """Validates single-use token and transitions account to ACTIVE status."""
        if not token:
            raise ValidationException("Verification token is required.")

        t_hash = hash_token(token)
        now = datetime.utcnow()

        verif = db.execute(
            select(EmailVerificationToken).where(
                and_(
                    EmailVerificationToken.token_hash == t_hash,
                    EmailVerificationToken.is_used == False
                )
            )
        ).scalar_one_or_none()

        if not verif or verif.expires_at < now:
            raise AuthenticationException(
                "Invalid or expired email verification token.",
                code="INVALID_VERIFICATION_TOKEN",
                status_code=400
            )

        # Mark token as used
        verif.is_used = True
        verif.used_at = now

        # Update User
        user = db.execute(select(User).where(User.id == verif.user_id)).scalar_one_or_none()
        if user:
            user.is_verified = True
            user.email_verified_at = now
            if user.account_status == AccountStatus.PENDING_VERIFICATION:
                user.account_status = AccountStatus.ACTIVE

            audit = AuditEvent(
                entity_name="User",
                entity_id=user.id,
                action="USER_EMAIL_VERIFIED",
                actor_id=user.id
            )
            db.add(audit)

        db.commit()
        return True

    @staticmethod
    def resend_verification(db: Session, email: str) -> bool:
        """Generates a fresh verification token for unverified accounts."""
        clean_email = normalize_email(email)
        user = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        
        # Avoid account enumeration: always return True regardless of user existence
        if not user or user.is_verified:
            return True

        now = datetime.utcnow()
        # Invalidate existing unused tokens
        db.execute(
            update(EmailVerificationToken)
            .where(EmailVerificationToken.user_id == user.id)
            .values(is_used=True, used_at=now)
        )

        raw_token = generate_secure_token(32)
        verif_token = EmailVerificationToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=now + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS)
        )
        db.add(verif_token)
        db.commit()

        notification_service.send_email_sync(
            recipient_email=clean_email,
            subject="Verify Your CharityChain Account",
            html_body=f"<p>Your new CharityChain verification token is: <b>{raw_token}</b></p>"
        )
        return True

    @staticmethod
    def authenticate_user(
        db: Session,
        req: LoginRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Authenticates credentials with rate/lockout protections, email verification checks,
        and establishes persistent session tracking.
        """
        clean_email = normalize_email(req.email)
        user = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        now = datetime.utcnow()

        if not user:
            # Constant-time mitigation against email enumeration
            verify_password("dummy-password", "$argon2id$v=19$m=65536,t=3,p=4$dummy$dummy")
            raise AuthenticationException("Invalid email or password.", code="INVALID_CREDENTIALS", status_code=401)

        # 1. Check Account Lockout
        if user.locked_until and user.locked_until > now:
            remaining = int((user.locked_until - now).total_seconds() / 60) + 1
            raise AuthenticationException(
                f"Account is temporarily locked due to repeated failed logins. Try again in {remaining} minutes.",
                code="ACCOUNT_LOCKED",
                status_code=423
            )

        # 2. Check Disabled/Suspended Status
        if user.account_status in (AccountStatus.SUSPENDED, AccountStatus.DEACTIVATED):
            raise ForbiddenException("Account has been disabled or suspended. Contact support.")

        # 3. Verify Password Hash
        if not verify_password(req.password, user.hashed_password):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= settings.MAX_FAILED_LOGIN_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES)
                logger.warning(f"Account <{user.email}> temporarily locked for {settings.ACCOUNT_LOCKOUT_MINUTES}m.")
            db.commit()
            raise AuthenticationException("Invalid email or password.", code="INVALID_CREDENTIALS", status_code=401)

        # 4. Check Email Verification Status
        if not user.is_verified or user.account_status == AccountStatus.PENDING_VERIFICATION:
            raise ForbiddenException("Email address is not verified. Please verify your email before logging in.")

        # 5. Reset Failed Attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = now

        # 6. Establish Persistent Session & Generate Token Pair
        raw_refresh_token = generate_secure_token(48)
        refresh_hash = hash_token(raw_refresh_token)
        token_family = str(uuid.uuid4())

        session = UserSession(
            user_id=user.id,
            refresh_token_hash=refresh_hash,
            token_family=token_family,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            last_used_at=now
        )
        db.add(session)
        db.flush()

        access_token = create_access_token(
            subject=user.id,
            platform_role=user.role.value,
            session_id=session.id
        )

        audit = AuditEvent(
            entity_name="UserSession",
            entity_id=session.id,
            action="USER_LOGIN_SUCCESS",
            actor_id=user.id,
            after_state={"ip_address": ip_address}
        )
        db.add(audit)
        db.commit()

        return TokenResponse(
            access_token=access_token,
            refresh_token=raw_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role
        )

    @staticmethod
    def rotate_refresh_token(
        db: Session,
        raw_refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Rotates refresh token. Detects token reuse and revokes entire family if compromise occurs.
        """
        if not raw_refresh_token:
            raise AuthenticationException("Refresh token is required.", code="MISSING_TOKEN", status_code=401)

        token_hash = hash_token(raw_refresh_token)
        now = datetime.utcnow()

        # Lookup session by token hash
        session = db.execute(
            select(UserSession).where(UserSession.refresh_token_hash == token_hash)
        ).scalar_one_or_none()

        if not session:
            raise AuthenticationException("Invalid or unrecognized refresh token.", code="INVALID_REFRESH_TOKEN", status_code=401)

        # REUSE DETECTION: If token was already revoked/rotated, this is an attempted token reuse!
        if session.is_revoked:
            logger.critical(
                f"SECURITY ALERT: Refresh token reuse detected for user <{session.user_id}>, family <{session.token_family}>!"
            )
            # Invalidate all sessions in this token family
            db.execute(
                update(UserSession)
                .where(UserSession.token_family == session.token_family)
                .values(is_revoked=True, revoked_at=now, revocation_reason="TOKEN_REUSE_DETECTED")
            )
            audit = AuditEvent(
                entity_name="UserSession",
                entity_id=session.id,
                action="TOKEN_REUSE_DETECTED",
                actor_id=session.user_id,
                after_state={"revoked_family": session.token_family}
            )
            db.add(audit)
            db.commit()
            raise AuthenticationException(
                "Compromised refresh token detected. All associated sessions have been revoked. Please log in again.",
                code="TOKEN_COMPROMISED",
                status_code=401
            )

        # Check Expiration
        if session.expires_at < now:
            session.is_revoked = True
            session.revoked_at = now
            session.revocation_reason = "EXPIRED"
            db.commit()
            raise TokenExpiredException("Refresh token has expired.")

        # Invalidate current token (Rotation)
        session.is_revoked = True
        session.revoked_at = now
        session.revocation_reason = "ROTATED"
        session.last_used_at = now

        # Fetch User
        user = db.execute(select(User).where(User.id == session.user_id)).scalar_one_or_none()
        if not user or user.account_status in (AccountStatus.SUSPENDED, AccountStatus.DEACTIVATED):
            db.commit()
            raise ForbiddenException("Account is disabled or suspended.")

        # Issue New Refresh Token in the same Token Family
        new_raw_refresh = generate_secure_token(48)
        new_refresh_hash = hash_token(new_raw_refresh)

        new_session = UserSession(
            user_id=user.id,
            refresh_token_hash=new_refresh_hash,
            token_family=session.token_family,
            ip_address=ip_address or session.ip_address,
            user_agent=user_agent or session.user_agent,
            expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            last_used_at=now
        )
        db.add(new_session)
        db.flush()

        access_token = create_access_token(
            subject=user.id,
            platform_role=user.role.value,
            session_id=new_session.id
        )

        db.commit()

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_raw_refresh,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role
        )

    @staticmethod
    def logout(db: Session, session_id: str, user_id: str) -> bool:
        """Revokes the current session."""
        session = db.execute(
            select(UserSession).where(
                and_(UserSession.id == session_id, UserSession.user_id == user_id)
            )
        ).scalar_one_or_none()

        if session and not session.is_revoked:
            session.is_revoked = True
            session.revoked_at = datetime.utcnow()
            session.revocation_reason = "USER_LOGOUT"
            db.commit()
        return True

    @staticmethod
    def logout_all_sessions(db: Session, user_id: str) -> bool:
        """Revokes all active sessions for the user."""
        now = datetime.utcnow()
        db.execute(
            update(UserSession)
            .where(and_(UserSession.user_id == user_id, UserSession.is_revoked == False))
            .values(is_revoked=True, revoked_at=now, revocation_reason="USER_LOGOUT_ALL")
        )
        db.commit()
        return True

    @staticmethod
    def initiate_password_reset(db: Session, email: str) -> bool:
        """Dispatches password reset recovery token. Fails safely without leaking user existence."""
        clean_email = normalize_email(email)
        user = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        
        if not user or user.account_status in (AccountStatus.SUSPENDED, AccountStatus.DEACTIVATED):
            return True

        now = datetime.utcnow()
        # Invalidate old unused reset tokens
        db.execute(
            update(PasswordResetToken)
            .where(PasswordResetToken.user_id == user.id)
            .values(is_used=True, used_at=now)
        )

        raw_token = generate_secure_token(32)
        reset_token = PasswordResetToken(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=now + timedelta(minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES)
        )
        db.add(reset_token)
        db.commit()

        notification_service.send_email_sync(
            recipient_email=clean_email,
            subject="CharityChain Password Reset Request",
            html_body=f"<p>Use this token to reset your password: <b>{raw_token}</b></p>"
        )
        return True

    @staticmethod
    def reset_password(db: Session, token: str, new_password: str, new_password_confirm: str) -> bool:
        """Resets password, invalidates token, and revokes all active sessions."""
        if new_password != new_password_confirm:
            raise ValidationException("Passwords do not match.", details={"field": "new_password_confirm"})

        valid_pwd, pwd_error = validate_password_strength(new_password)
        if not valid_pwd:
            raise ValidationException(pwd_error or "Password validation failed.")

        token_hash = hash_token(token)
        now = datetime.utcnow()

        token_record = db.execute(
            select(PasswordResetToken).where(
                and_(
                    PasswordResetToken.token_hash == token_hash,
                    PasswordResetToken.is_used == False
                )
            )
        ).scalar_one_or_none()

        if not token_record or token_record.expires_at < now:
            raise AuthenticationException("Invalid or expired password reset token.", code="INVALID_RESET_TOKEN", status_code=400)

        # Mark token used
        token_record.is_used = True
        token_record.used_at = now

        # Update User Password
        user = db.execute(select(User).where(User.id == token_record.user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        user.hashed_password = hash_password(new_password)
        user.password_changed_at = now

        # Security: Revoke all active sessions upon password reset!
        db.execute(
            update(UserSession)
            .where(and_(UserSession.user_id == user.id, UserSession.is_revoked == False))
            .values(is_revoked=True, revoked_at=now, revocation_reason="PASSWORD_RESET")
        )

        audit = AuditEvent(
            entity_name="User",
            entity_id=user.id,
            action="USER_PASSWORD_RESET",
            actor_id=user.id
        )
        db.add(audit)
        db.commit()
        return True

    @staticmethod
    def change_password(db: Session, user_id: str, current_password: str, new_password: str, new_password_confirm: str) -> bool:
        """Authenticated password change. Invalidates other sessions."""
        if new_password != new_password_confirm:
            raise ValidationException("New passwords do not match.")

        valid_pwd, pwd_error = validate_password_strength(new_password)
        if not valid_pwd:
            raise ValidationException(pwd_error or "Password validation failed.")

        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationException("Current password is incorrect.", code="INCORRECT_PASSWORD", status_code=400)

        now = datetime.utcnow()
        user.hashed_password = hash_password(new_password)
        user.password_changed_at = now

        # Revoke other sessions
        db.execute(
            update(UserSession)
            .where(and_(UserSession.user_id == user.id, UserSession.is_revoked == False))
            .values(is_revoked=True, revoked_at=now, revocation_reason="PASSWORD_CHANGED")
        )

        audit = AuditEvent(
            entity_name="User",
            entity_id=user.id,
            action="USER_PASSWORD_CHANGED",
            actor_id=user.id
        )
        db.add(audit)
        db.commit()
        return True

