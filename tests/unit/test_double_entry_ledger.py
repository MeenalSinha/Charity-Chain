"""
test_double_entry_ledger.py - Double-Entry Ledger & Invariant Enforcement Unit Tests
===================================================================================
Tests double-entry posting rules:
1. Invariant: Sum(Debits) == Sum(Credits) strictly enforced.
2. Unbalanced entries are rejected with ValidationException.
3. Negative monetary values are rejected.
4. Trial balance is always balanced.
5. Reversal entries maintain exact mathematical parity.
"""

import pytest
import uuid
from backend.app.core.constants import (
    LedgerAccountType, LedgerEntryType, PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus
)
from backend.app.core.exceptions import ValidationException, ConflictException
from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund, Disbursement
)
from backend.app.services.ledger_service import LedgerService


def test_unbalanced_journal_entry_rejected(db_session):
    """Ensures an entry where debits != credits fails with ValidationException."""
    lines = [
        {"account_number": "1000", "debit_minor_units": 10000, "credit_minor_units": 0},
        {"account_number": "2000", "debit_minor_units": 0, "credit_minor_units": 8000}  # $20.00 discrepancy
    ]

    with pytest.raises(ValidationException) as exc:
        LedgerService.post_journal_entry(
            db=db_session,
            entry_type=LedgerEntryType.MANUAL_ADJUSTMENT,
            description="Unbalanced test entry",
            lines=lines
        )
    assert "Double-entry ledger invariant violation" in str(exc.value)


def test_negative_minor_units_rejected(db_session):
    """Ensures negative debits or credits are rejected."""
    lines = [
        {"account_number": "1000", "debit_minor_units": -5000, "credit_minor_units": 0},
        {"account_number": "2000", "debit_minor_units": 0, "credit_minor_units": -5000}
    ]

    with pytest.raises(ValidationException) as exc:
        LedgerService.post_journal_entry(
            db=db_session,
            entry_type=LedgerEntryType.MANUAL_ADJUSTMENT,
            description="Negative test entry",
            lines=lines
        )
    assert "Negative monetary amounts are not permitted" in str(exc.value)


def test_balanced_donation_and_fee_posting(db_session):
    """Tests posting a donation capture with processor fee and verifies trial balance."""
    don = Donation(id=str(uuid.uuid4()), project_id="proj-1", milestone_id="ms-1", amount_minor_units=50000)
    cap = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-TEST-1", amount_minor_units=50000, currency="USD", status=PaymentCaptureStatus.COMPLETED)

    # Gross: $500.00 (50,000 cents), Fee: $15.00 (1,500 cents)
    entry = LedgerService.record_donation_capture(
        db=db_session,
        donation=don,
        capture=cap,
        processor_fee_minor_units=1500
    )

    assert entry.is_reversed is False
    assert len(entry.lines) == 4

    # Verify trial balance
    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True
    assert tb["discrepancy_minor_units"] == 0
    assert tb["total_debit_minor_units"] == 51500
    assert tb["total_credit_minor_units"] == 51500


def test_payout_commitment_and_execution_lifecycle(db_session):
    """Tests ledger postings: commitment -> execution -> trial balance verification."""
    disb = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id="ms-100",
        organization_id="org-100",
        amount_minor_units=20000,
        currency="USD",
        status=DisbursementStatus.APPROVED
    )

    # 1. Commitment posting
    comm_entry = LedgerService.record_payout_commitment(db=db_session, disbursement=disb)
    assert comm_entry.entry_type == LedgerEntryType.PAYOUT_COMMITMENT

    tb1 = LedgerService.get_trial_balance(db_session)
    assert tb1["is_balanced"] is True

    # 2. Execution posting
    exec_entry = LedgerService.record_payout_execution(
        db=db_session,
        disbursement=disb,
        provider_fee_minor_units=25  # $0.25 fee
    )
    assert exec_entry.entry_type == LedgerEntryType.PAYOUT_EXECUTION

    tb2 = LedgerService.get_trial_balance(db_session)
    assert tb2["is_balanced"] is True


def test_journal_entry_reversal_parity(db_session):
    """Verifies that reversing an entry inverts lines and preserves trial balance parity."""
    don = Donation(id=str(uuid.uuid4()), project_id="proj-2", milestone_id="ms-2", amount_minor_units=10000)
    cap = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-REV-1", amount_minor_units=10000, currency="USD", status=PaymentCaptureStatus.COMPLETED)

    entry = LedgerService.record_donation_capture(db=db_session, donation=don, capture=cap)

    reversal = LedgerService.reverse_journal_entry(
        db=db_session,
        entry_id=entry.id,
        reason="Erroneous charge processed"
    )

    db_session.refresh(entry)
    assert entry.is_reversed is True
    assert entry.reversed_by_entry_id == reversal.id

    # Second reversal of same entry must fail
    with pytest.raises(ConflictException):
        LedgerService.reverse_journal_entry(db=db_session, entry_id=entry.id, reason="Double reversal")

    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True
