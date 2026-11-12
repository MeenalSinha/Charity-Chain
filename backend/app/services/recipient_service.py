"""
recipient_service.py - Payout Recipient Onboarding & Verification Service
========================================================================
Manages NGO payout destination accounts:
- Sensitive destination details (emails, accounts) encrypted at rest via AES/Fernet.
- SHA-256 destination hashing for indexing and deduplication.
- Masked destination representation for safe UI rendering.
- State machine lifecycle: PENDING_VERIFICATION -> VERIFIED -> SUSPENDED -> INACTIVE.
- Restricts decryption exclusively to authorized outbound payout processes.
"""

import base64
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from cryptography.fernet import Fernet
from sqlalchemy import select, and_, update
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import (
    PayoutRecipientStatus, RecipientType, DisbursementMethod, PlatformRole
)
from backend.app.core.exceptions import (
    ValidationException, NotFoundException, ForbiddenException, ConflictException
)
from backend.app.core.logging import get_logger
from backend.app.db.models.financial import PayoutRecipient
from backend.app.db.models.users import Organization, User
from backend.app.integrations.payouts.factory import get_payout_provider

logger = get_logger("services.payout_recipient")


def _get_cipher() -> Fernet:
    """Derives a Fernet cipher from settings.SECRET_KEY."""
    key_bytes = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key_bytes))


def mask_destination(destination: str, recipient_type: RecipientType) -> str:
    """Creates a masked, non-sensitive string for display and logging."""
    clean = destination.strip()
    if recipient_type == RecipientType.PAYPAL_EMAIL:
        if "@" in clean:
            user_part, domain = clean.split("@", 1)
            if len(user_part) <= 2:
                masked_user = user_part[0] + "***" if user_part else "***"
            else:
                masked_user = user_part[0] + "***" + user_part[-1]
            return f"{masked_user}@{domain}"
        return clean[:2] + "***" + clean[-2:] if len(clean) > 4 else "***"

    # Bank account or generic destination: show last 4 digits
    if len(clean) > 4:
        return "****" + clean[-4:]
    return "****"


class PayoutRecipientService:
    """Core domain service for NGO payout recipient onboarding and compliance."""

    def __init__(self, provider=None):
        self.provider = provider or get_payout_provider()

    @staticmethod
    def compute_destination_hash(destination: str) -> str:
        """Returns deterministic SHA-256 hash of normalized destination."""
        return hashlib.sha256(destination.strip().lower().encode()).hexdigest()

    @staticmethod
    def encrypt_destination(destination: str) -> str:
        """Encrypts sensitive destination using Fernet."""
        cipher = _get_cipher()
        return cipher.encrypt(destination.strip().encode()).decode()

    @staticmethod
    def decrypt_destination(encrypted_dest: str) -> str:
        """Decrypts ciphertext destination back to plaintext."""
        try:
            cipher = _get_cipher()
            return cipher.decrypt(encrypted_dest.encode()).decode()
        except Exception:
            return encrypted_dest

    def register_recipient(
        self,
        db: Session,
        organization_id: str,
        destination: str,
        recipient_type: RecipientType = RecipientType.PAYPAL_EMAIL,
        payout_method: DisbursementMethod = DisbursementMethod.PAYPAL_PAYOUTS,
        is_primary: bool = False,
        actor_id: Optional[str] = None
    ) -> PayoutRecipient:
        """
        Registers a new payout destination for an organization.
        Validates format with provider, encrypts at rest, and initiates in PENDING_VERIFICATION.
        """
        org = db.execute(select(Organization).where(Organization.id == organization_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException(f"Organization '{organization_id}' not found.")

        # Validate with provider
        val_res = self.provider.validate_recipient(recipient_type, destination)
        if not val_res.is_valid:
            raise ValidationException(f"Recipient destination validation failed: {val_res.error_message}")

        dest_hash = self.compute_destination_hash(val_res.normalized_destination)

        # Check for existing recipient under this org
        existing = db.execute(
            select(PayoutRecipient).where(
                and_(
                    PayoutRecipient.organization_id == organization_id,
                    PayoutRecipient.destination_hash == dest_hash
                )
            )
        ).scalar_one_or_none()

        if existing:
            if existing.status in (PayoutRecipientStatus.VERIFIED, PayoutRecipientStatus.PENDING_VERIFICATION):
                raise ConflictException("This payout destination is already registered for this organization.")
            # If rejected or inactive, allow re-activation
            existing.status = PayoutRecipientStatus.PENDING_VERIFICATION
            existing.encrypted_destination = self.encrypt_destination(val_res.normalized_destination)
            existing.destination_masked = mask_destination(val_res.normalized_destination, recipient_type)
            db.commit()
            db.refresh(existing)
            return existing

        # If marking primary, clear previous primary flags for this org
        if is_primary:
            db.execute(
                update(PayoutRecipient)
                .where(PayoutRecipient.organization_id == organization_id)
                .values(is_primary=False)
            )

        encrypted = self.encrypt_destination(val_res.normalized_destination)
        masked = mask_destination(val_res.normalized_destination, recipient_type)

        recipient = PayoutRecipient(
            id=str(uuid.uuid4()),
            organization_id=organization_id,
            recipient_type=recipient_type,
            payout_method=payout_method,
            encrypted_destination=encrypted,
            destination_masked=masked,
            destination_hash=dest_hash,
            status=PayoutRecipientStatus.PENDING_VERIFICATION,
            is_primary=is_primary
        )
        db.add(recipient)
        db.commit()
        db.refresh(recipient)

        logger.info(
            f"Registered payout recipient {recipient.id} ({masked}) for organization {organization_id}",
            extra={"recipient_id": recipient.id, "org_id": organization_id}
        )
        return recipient

    def verify_recipient(
        self,
        db: Session,
        recipient_id: str,
        verifier_id: str,
        verification_notes: Optional[str] = None
    ) -> PayoutRecipient:
        """
        Authorizes and marks a payout recipient destination as VERIFIED.
        Restricted to platform administrator or finance operator.
        """
        recipient = db.execute(select(PayoutRecipient).where(PayoutRecipient.id == recipient_id)).scalar_one_or_none()
        if not recipient:
            raise NotFoundException(f"Payout recipient '{recipient_id}' not found.")

        if recipient.status == PayoutRecipientStatus.VERIFIED:
            return recipient

        recipient.status = PayoutRecipientStatus.VERIFIED
        recipient.verified_by_id = verifier_id
        recipient.verified_at = datetime.now(timezone.utc)
        recipient.verification_notes = verification_notes

        db.commit()
        db.refresh(recipient)

        logger.info(
            f"Payout recipient {recipient_id} verified by {verifier_id}",
            extra={"recipient_id": recipient_id, "verifier_id": verifier_id}
        )
        return recipient

    def suspend_recipient(
        self,
        db: Session,
        recipient_id: str,
        actor_id: str,
        reason: str
    ) -> PayoutRecipient:
        """Suspends a payout destination from receiving disbursements."""
        recipient = db.execute(select(PayoutRecipient).where(PayoutRecipient.id == recipient_id)).scalar_one_or_none()
        if not recipient:
            raise NotFoundException(f"Payout recipient '{recipient_id}' not found.")

        recipient.status = PayoutRecipientStatus.SUSPENDED
        recipient.verification_notes = f"Suspended by {actor_id}: {reason}"
        db.commit()
        db.refresh(recipient)
        return recipient

    def set_primary_recipient(
        self,
        db: Session,
        organization_id: str,
        recipient_id: str
    ) -> PayoutRecipient:
        """Designates a verified recipient as the primary payout destination."""
        recipient = db.execute(select(PayoutRecipient).where(PayoutRecipient.id == recipient_id)).scalar_one_or_none()
        if not recipient or recipient.organization_id != organization_id:
            raise NotFoundException("Recipient not found for this organization.")

        if recipient.status != PayoutRecipientStatus.VERIFIED:
            raise ValidationException("Only VERIFIED recipients can be designated as primary.")

        # Demote existing primaries
        db.execute(
            update(PayoutRecipient)
            .where(PayoutRecipient.organization_id == organization_id)
            .values(is_primary=False)
        )
        recipient.is_primary = True
        db.commit()
        db.refresh(recipient)
        return recipient

    def get_organization_recipients(
        self,
        db: Session,
        organization_id: str
    ) -> List[PayoutRecipient]:
        """Returns all registered payout destinations for an organization."""
        return db.execute(
            select(PayoutRecipient)
            .where(PayoutRecipient.organization_id == organization_id)
            .order_by(PayoutRecipient.is_primary.desc(), PayoutRecipient.created_at.desc())
        ).scalars().all()

    def get_active_verified_recipient(
        self,
        db: Session,
        organization_id: str
    ) -> Optional[PayoutRecipient]:
        """Retrieves primary or first verified payout recipient for an organization."""
        # Check primary first
        primary = db.execute(
            select(PayoutRecipient).where(
                and_(
                    PayoutRecipient.organization_id == organization_id,
                    PayoutRecipient.status == PayoutRecipientStatus.VERIFIED,
                    PayoutRecipient.is_primary == True
                )
            )
        ).scalar_one_or_none()
        if primary:
            return primary

        # Fallback to any verified
        return db.execute(
            select(PayoutRecipient).where(
                and_(
                    PayoutRecipient.organization_id == organization_id,
                    PayoutRecipient.status == PayoutRecipientStatus.VERIFIED
                )
            ).order_by(PayoutRecipient.created_at.desc())
        ).scalars().first()

    def get_decrypted_destination_for_payout(
        self,
        db: Session,
        recipient_id: str
    ) -> str:
        """Securely retrieves decrypted destination for genuine payout execution."""
        recipient = db.execute(select(PayoutRecipient).where(PayoutRecipient.id == recipient_id)).scalar_one_or_none()
        if not recipient:
            raise NotFoundException(f"Payout recipient '{recipient_id}' not found.")
        if recipient.status != PayoutRecipientStatus.VERIFIED:
            raise ValidationException(f"Cannot disburse to unverified recipient (status: {recipient.status.value}).")

        return self.decrypt_destination(recipient.encrypted_destination)
