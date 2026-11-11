"""
admin_service.py - Platform Administrative Identity & Access Service
=====================================================================
Administrative controls for user listing, account suspension, session revocation,
platform role assignment, and security audit event inspection.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, func, update, and_, or_

from backend.app.core.constants import PlatformRole, AccountStatus
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.db.models.users import User, UserProfile, UserSession
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.api.schemas import UserResponse, UserProfileResponse


class AdminService:
    """Platform administrative operations."""

    @staticmethod
    def list_users(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        status: Optional[AccountStatus] = None,
        role: Optional[PlatformRole] = None,
        search: Optional[str] = None
    ) -> Tuple[List[UserResponse], int]:
        """Returns paginated users with filtering and search."""
        query = select(User)
        count_query = select(func.count(User.id))

        filters = []
        if status:
            filters.append(User.account_status == status)
        if role:
            filters.append(User.role == role)
        if search:
            s = f"%{search.strip().lower()}%"
            filters.append(or_(User.email.ilike(s), User.full_name.ilike(s)))

        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))

        total = db.execute(count_query).scalar_one()

        offset = (page - 1) * page_size
        users = db.execute(
            query.order_by(User.created_at.desc()).offset(offset).limit(page_size)
        ).scalars().all()

        results = []
        for u in users:
            prof_dto = None
            if u.profile:
                prof_dto = UserProfileResponse(
                    bio=u.profile.bio,
                    phone_number=u.profile.phone_number,
                    avatar_url=u.profile.avatar_url
                )
            results.append(
                UserResponse(
                    id=u.id,
                    email=u.email,
                    full_name=u.full_name,
                    role=u.role,
                    account_status=u.account_status,
                    is_verified=u.is_verified,
                    email_verified_at=u.email_verified_at,
                    created_at=u.created_at,
                    profile=prof_dto
                )
            )

        return results, total

    @staticmethod
    def get_user_details(db: Session, user_id: str) -> UserResponse:
        """Retrieves user profile and account details."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        prof_dto = None
        if user.profile:
            prof_dto = UserProfileResponse(
                bio=user.profile.bio,
                phone_number=user.profile.phone_number,
                avatar_url=user.profile.avatar_url
            )

        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            account_status=user.account_status,
            is_verified=user.is_verified,
            email_verified_at=user.email_verified_at,
            created_at=user.created_at,
            profile=prof_dto
        )

    @staticmethod
    def update_user_status(db: Session, user_id: str, new_status: AccountStatus, admin_id: str) -> bool:
        """Updates user status. If suspended or deactivated, revokes all active sessions."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        old_status = user.account_status
        user.account_status = new_status
        now = datetime.utcnow()

        # If suspended or deactivated, immediately revoke all active sessions!
        if new_status in (AccountStatus.SUSPENDED, AccountStatus.DEACTIVATED):
            db.execute(
                update(UserSession)
                .where(and_(UserSession.user_id == user.id, UserSession.is_revoked == False))
                .values(is_revoked=True, revoked_at=now, revocation_reason=f"ADMIN_ACCOUNT_{new_status.value}")
            )

        audit = AuditEvent(
            entity_name="User",
            entity_id=user.id,
            action="ADMIN_UPDATE_USER_STATUS",
            actor_id=admin_id,
            before_state={"status": old_status.value},
            after_state={"status": new_status.value}
        )
        db.add(audit)
        db.commit()
        return True

    @staticmethod
    def update_user_role(db: Session, user_id: str, new_role: PlatformRole, admin_id: str) -> bool:
        """Assigns a new platform role (DONOR, REVIEWER, PLATFORM_ADMIN) to a user."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("User not found.")

        old_role = user.role
        user.role = new_role

        audit = AuditEvent(
            entity_name="User",
            entity_id=user.id,
            action="ADMIN_UPDATE_USER_ROLE",
            actor_id=admin_id,
            before_state={"role": old_role.value},
            after_state={"role": new_role.value}
        )
        db.add(audit)
        db.commit()
        return True

    @staticmethod
    def list_security_events(
        db: Session,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lists audit events for security and administrative review."""
        query = select(AuditEvent)
        count_query = select(func.count(AuditEvent.id))

        if action:
            query = query.where(AuditEvent.action == action)
            count_query = count_query.where(AuditEvent.action == action)

        total = db.execute(count_query).scalar_one()

        offset = (page - 1) * page_size
        events = db.execute(
            query.order_by(AuditEvent.created_at.desc()).offset(offset).limit(page_size)
        ).scalars().all()

        results = [
            {
                "id": e.id,
                "entity_name": e.entity_name,
                "entity_id": e.entity_id,
                "action": e.action,
                "actor_id": e.actor_id,
                "created_at": e.created_at.isoformat(),
                "after_state": e.after_state,
                "correlation_id": e.correlation_id
            }
            for e in events
        ]
        return results, total

