"""
test_payout_recipient.py - Payout Recipient Onboarding & Encryption Unit Tests
==============================================================================
Tests recipient destination validation, Fernet encryption at rest, masking,
SHA-256 deduplication, lifecycle transitions, and primary designation.
"""

import pytest
import uuid
from backend.app.core.constants import (
    OrganizationStatus, PayoutRecipientStatus, RecipientType, DisbursementMethod
)
from backend.app.core.exceptions import ValidationException, ConflictException
from backend.app.db.models.users import Organization
from backend.app.services.recipient_service import (
    PayoutRecipientService, mask_destination
)
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def recipient_org_setup(db_session):
    """Sets up an organization for recipient tests."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Waters Global",
        legal_name="Clean Waters Global NGO",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)
    db_session.commit()
    return org


def test_destination_masking():
    """Verifies privacy masking for email and bank destinations."""
    assert mask_destination("finance@cleanwaters.org", RecipientType.PAYPAL_EMAIL) == "f***e@cleanwaters.org"
    assert mask_destination("ab@charity.org", RecipientType.PAYPAL_EMAIL) == "a***@charity.org"
    assert mask_destination("1234567890", RecipientType.BANK_ACCOUNT) == "****7890"


def test_recipient_registration_and_encryption(db_session, recipient_org_setup):
    """Verifies recipient is stored encrypted, masked, and in PENDING_VERIFICATION."""
    org = recipient_org_setup
    mock_provider = MockPayoutProvider()
    srv = PayoutRecipientService(provider=mock_provider)

    raw_email = "Disbursements@CleanWaters.org"
    recip = srv.register_recipient(
        db=db_session,
        organization_id=org.id,
        destination=raw_email,
        recipient_type=RecipientType.PAYPAL_EMAIL,
        is_primary=True
    )

    assert recip.status == PayoutRecipientStatus.PENDING_VERIFICATION
    assert recip.is_primary is True
    assert recip.destination_masked == "d***s@cleanwaters.org"
    # Ensure plaintext is NEVER stored in database
    assert raw_email not in recip.encrypted_destination
    assert "@" not in recip.encrypted_destination

    # Decrypt and verify
    decrypted = srv.decrypt_destination(recip.encrypted_destination)
    assert decrypted == raw_email.lower()


def test_duplicate_recipient_conflict(db_session, recipient_org_setup):
    """Prevents registering identical destination twice for same organization."""
    org = recipient_org_setup
    mock_provider = MockPayoutProvider()
    srv = PayoutRecipientService(provider=mock_provider)

    srv.register_recipient(
        db=db_session,
        organization_id=org.id,
        destination="payouts@cleanwaters.org"
    )

    with pytest.raises(ConflictException):
        srv.register_recipient(
            db=db_session,
            organization_id=org.id,
            destination="  PAYOUTS@CleanWaters.org "  # Case and whitespace normalized
        )


def test_recipient_verification_lifecycle(db_session, recipient_org_setup):
    """Tests lifecycle: PENDING_VERIFICATION -> VERIFIED -> SUSPENDED."""
    org = recipient_org_setup
    mock_provider = MockPayoutProvider()
    srv = PayoutRecipientService(provider=mock_provider)

    recip = srv.register_recipient(
        db=db_session,
        organization_id=org.id,
        destination="ops@cleanwaters.org"
    )
    assert recip.status == PayoutRecipientStatus.PENDING_VERIFICATION

    # Verify
    verified = srv.verify_recipient(
        db=db_session,
        recipient_id=recip.id,
        verifier_id="admin-001",
        verification_notes="Verified via corporate registration documents"
    )
    assert verified.status == PayoutRecipientStatus.VERIFIED
    assert verified.verified_by_id == "admin-001"
    assert verified.verified_at is not None

    # Suspend
    suspended = srv.suspend_recipient(
        db=db_session,
        recipient_id=recip.id,
        actor_id="admin-001",
        reason="Suspicious activity reported"
    )
    assert suspended.status == PayoutRecipientStatus.SUSPENDED


def test_primary_recipient_switching(db_session, recipient_org_setup):
    """Ensures designating a new primary recipient unsets previous primary."""
    org = recipient_org_setup
    mock_provider = MockPayoutProvider()
    srv = PayoutRecipientService(provider=mock_provider)

    r1 = srv.register_recipient(db=db_session, organization_id=org.id, destination="primary1@ngo.org", is_primary=True)
    r2 = srv.register_recipient(db=db_session, organization_id=org.id, destination="primary2@ngo.org", is_primary=False)

    srv.verify_recipient(db=db_session, recipient_id=r1.id, verifier_id="admin")
    srv.verify_recipient(db=db_session, recipient_id=r2.id, verifier_id="admin")

    # Set r2 as primary
    srv.set_primary_recipient(db=db_session, organization_id=org.id, recipient_id=r2.id)

    db_session.refresh(r1)
    db_session.refresh(r2)
    assert r1.is_primary is False
    assert r2.is_primary is True
