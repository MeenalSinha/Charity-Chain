"""
cli.py - Administrative CLI Tools
==================================
Command line interface for secure administrative operations, including
bootstrapping the initial platform administrator account.

Usage:
    python -m backend.app.cli create-platform-admin --email admin@charitychain.org --name "Platform Admin"
"""

import sys
import argparse
from datetime import datetime
from sqlalchemy import select

from backend.app.core.constants import PlatformRole, AccountStatus
from backend.app.core.security import hash_password, normalize_email, validate_password_strength
from backend.app.db.session import SessionFactory
from backend.app.db.models.users import User, UserProfile
from backend.app.db.models.reviews_impact import AuditEvent


def create_platform_admin(email: str, name: str, password: str) -> bool:
    """Creates a new platform administrator account with validated input and audit record."""
    clean_email = normalize_email(email)
    if not clean_email or "@" not in clean_email:
        print(f"Error: Invalid email address '{email}'.")
        return False

    valid_pwd, pwd_error = validate_password_strength(password)
    if not valid_pwd:
        print(f"Error: {pwd_error}")
        return False

    with SessionFactory() as db:
        existing = db.execute(select(User).where(User.email == clean_email)).scalar_one_or_none()
        if existing:
            if existing.role == PlatformRole.PLATFORM_ADMIN:
                print(f"User '{clean_email}' already exists and is already a PLATFORM_ADMIN.")
                return True
            print(f"User '{clean_email}' already exists. Elevating role to PLATFORM_ADMIN.")
            existing.role = PlatformRole.PLATFORM_ADMIN
            existing.is_verified = True
            existing.account_status = AccountStatus.ACTIVE
            db.commit()
            print(f"Successfully elevated '{clean_email}' to PLATFORM_ADMIN.")
            return True

        hashed = hash_password(password)
        now = datetime.utcnow()
        admin_user = User(
            email=clean_email,
            full_name=name.strip(),
            hashed_password=hashed,
            role=PlatformRole.PLATFORM_ADMIN,
            account_status=AccountStatus.ACTIVE,
            is_verified=True,
            email_verified_at=now,
            password_changed_at=now
        )
        db.add(admin_user)
        db.flush()

        profile = UserProfile(user_id=admin_user.id, bio="Platform Administrator")
        db.add(profile)

        audit = AuditEvent(
            entity_name="User",
            entity_id=admin_user.id,
            action="PLATFORM_ADMIN_PROVISIONED_VIA_CLI",
            actor_id="CLI",
            after_state={"email": clean_email, "role": PlatformRole.PLATFORM_ADMIN.value}
        )
        db.add(audit)
        db.commit()

        print(f"Successfully created PLATFORM_ADMIN account for '{clean_email}' (ID: {admin_user.id}).")
        return True


def main():
    parser = argparse.ArgumentParser(description="CharityChain Administrative Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Administrative command to execute")

    admin_parser = subparsers.add_parser("create-platform-admin", help="Bootstrap a platform administrator account")
    admin_parser.add_argument("--email", required=True, help="Administrator email address")
    admin_parser.add_argument("--name", default="System Administrator", help="Administrator full name")
    admin_parser.add_argument("--password", required=True, help="Secure password (minimum 8 chars)")

    args = parser.parse_args()

    if args.command == "create-platform-admin":
        success = create_platform_admin(args.email, args.name, args.password)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

