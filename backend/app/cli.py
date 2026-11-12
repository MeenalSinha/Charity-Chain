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


def generate_issuer_key(key_id: str) -> bool:
    """Generates an Ed25519 issuer signing keypair and registers the public key."""
    from backend.app.services.key_management_service import KeyManagementService
    with SessionFactory() as db:
        new_kid, record = KeyManagementService.rotate_key(db=db, new_key_id=key_id)
        print(f"Successfully generated and registered issuer signing key '{new_kid}'.")
        print(f"Algorithm: {record.algorithm}")
        print(f"Status: {record.status}")
        return True


def verify_certificate_file(file_path: str) -> bool:
    """Independently verifies an offline certificate JSON payload."""
    import json
    import hashlib
    from pathlib import Path
    from backend.app.core.canonical_json import canonicalize_json
    from backend.app.services.key_management_service import KeyManagementService
    from backend.app.db.models.reviews_impact import IssuerSigningKey

    p = Path(file_path)
    if not p.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return False

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either raw certificate record or claims envelope
    claims = data.get("claims") or data.get("canonical_payload") or data
    if isinstance(claims, str):
        claims = json.loads(claims)

    signature_b64 = data.get("signature") or ""
    signing_key_id = data.get("signing_key_id") or claims.get("signing_key_id") or "key-charitychain-2026-v1"

    canonical_str = canonicalize_json(claims)
    payload_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    with SessionFactory() as db:
        key_record = db.query(IssuerSigningKey).filter(IssuerSigningKey.key_id == signing_key_id).first()
        key_pem = key_record.public_key_pem if key_record else ""

    if not key_pem:
        # Check active key in memory/file
        kid, priv = KeyManagementService.get_active_private_key()
        if kid == signing_key_id:
            from cryptography.hazmat.primitives import serialization
            key_pem = priv.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode("utf-8")

    if not key_pem:
        print(f"Error: Signing key '{signing_key_id}' public verification key not found.")
        return False

    is_sig_valid = KeyManagementService.verify_signature(canonical_str, signature_b64, key_pem)

    print("=" * 60)
    print("CHARITYCHAIN OFFLINE CERTIFICATE VERIFICATION REPORT")
    print("=" * 60)
    print(f"Certificate Number: {claims.get('certificate_number', 'N/A')}")
    print(f"Certificate Type:   {claims.get('certificate_type', 'N/A')}")
    print(f"Issuer:             {claims.get('issuer', {}).get('name', 'N/A')}")
    print(f"Project:            {claims.get('project', {}).get('title', 'N/A')}")
    print(f"Milestone:          {claims.get('milestone', {}).get('title', 'N/A')}")
    print(f"Signing Key ID:     {signing_key_id}")
    print(f"Payload SHA-256:    {payload_hash}")
    print(f"Signature Status:   {'[PASS] VALID' if is_sig_valid else '[FAIL] INVALID'}")
    print("=" * 60)
    return is_sig_valid


def main():
    parser = argparse.ArgumentParser(description="CharityChain Administrative Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Administrative command to execute")

    admin_parser = subparsers.add_parser("create-platform-admin", help="Bootstrap a platform administrator account")
    admin_parser.add_argument("--email", required=True, help="Administrator email address")
    admin_parser.add_argument("--name", default="System Administrator", help="Administrator full name")
    admin_parser.add_argument("--password", required=True, help="Secure password (minimum 8 chars)")

    key_parser = subparsers.add_parser("generate-issuer-key", help="Generate and register an Ed25519 issuer signing key")
    key_parser.add_argument("--key-id", required=True, help="Unique key identifier (e.g. key-charitychain-2026-v2)")

    verify_parser = subparsers.add_parser("verify-certificate", help="Independently verify an offline certificate JSON file")
    verify_parser.add_argument("file", help="Path to certificate JSON file")

    args = parser.parse_args()

    if args.command == "create-platform-admin":
        success = create_platform_admin(args.email, args.name, args.password)
        sys.exit(0 if success else 1)
    elif args.command == "generate-issuer-key":
        success = generate_issuer_key(args.key_id)
        sys.exit(0 if success else 1)
    elif args.command == "verify-certificate":
        success = verify_certificate_file(args.file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)



if __name__ == "__main__":
    main()

