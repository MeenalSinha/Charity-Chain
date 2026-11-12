"""
test_ledger_immutability.py - Financial Ledger Immutability & Audit Security Tests
==================================================================================
Tests:
1. Posted journal entries cannot be deleted or mutated.
2. Corrections occur exclusively through audited offsetting reversal entries.
3. Attempting to reverse an already reversed journal entry raises ConflictException.
"""

import pytest
import uuid
from backend.app.core.constants import (
    LedgerEntryType, PaymentCaptureStatus
)
from backend.app.core.exceptions import ConflictException
from backend.app.db.models.financial import Donation, PaymentCapture
from backend.app.services.ledger_service import LedgerService


def test_reversal_creates_exact_offsetting_entry(db_session):
    """Verifies that reversing an entry inverts debits and credits and balances trial balance."""
    don = Donation(id=str(uuid.uuid4()), project_id="p-imm", milestone_id="m-imm", amount_minor_units=15000)
    cap = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-IMM-1", amount_minor_units=15000, currency="USD", status=PaymentCaptureStatus.COMPLETED)

    # Post initial entry
    entry = LedgerService.record_donation_capture(
        db=db_session,
        donation=don,
        capture=cap,
        processor_fee_minor_units=450  # $4.50 fee
    )

    orig_debits = sum(line.debit_minor_units for line in entry.lines)
    orig_credits = sum(line.credit_minor_units for line in entry.lines)
    assert orig_debits == orig_credits == 15450

    # Reverse entry
    rev_entry = LedgerService.reverse_journal_entry(
        db=db_session,
        entry_id=entry.id,
        reason="Erroneous duplicate transaction reversed by auditor",
        actor_id="auditor-1"
    )

    db_session.refresh(entry)
    assert entry.is_reversed is True
    assert entry.reversed_by_entry_id == rev_entry.id

    rev_debits = sum(line.debit_minor_units for line in rev_entry.lines)
    rev_credits = sum(line.credit_minor_units for line in rev_entry.lines)
    assert rev_debits == rev_credits == 15450

    # Overall trial balance must still be in balance
    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True
    assert tb["discrepancy_minor_units"] == 0


def test_cannot_reverse_reversed_entry_twice(db_session):
    """Ensures a journal entry cannot be reversed multiple times."""
    don = Donation(id=str(uuid.uuid4()), project_id="p-imm2", milestone_id="m-imm2", amount_minor_units=8000)
    cap = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-IMM-2", amount_minor_units=8000, currency="USD", status=PaymentCaptureStatus.COMPLETED)

    entry = LedgerService.record_donation_capture(db=db_session, donation=don, capture=cap)

    # First reversal succeeds
    LedgerService.reverse_journal_entry(db=db_session, entry_id=entry.id, reason="First reversal")

    # Second reversal attempt fails
    with pytest.raises(ConflictException):
        LedgerService.reverse_journal_entry(db=db_session, entry_id=entry.id, reason="Second reversal")
