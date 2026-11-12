"""
test_payout_state_machine.py - Outbound Payout Lifecycle State Machine Tests
============================================================================
Tests allowed and forbidden transitions across DisbursementStatus.
"""

import pytest
from backend.app.core.constants import DisbursementStatus
from backend.app.core.exceptions import ValidationException


def validate_disbursement_transition(current: DisbursementStatus, target: DisbursementStatus) -> bool:
    """Validates allowed disbursement state machine transitions."""
    allowed = {
        DisbursementStatus.REQUESTED: {DisbursementStatus.PENDING_APPROVAL, DisbursementStatus.CANCELLED},
        DisbursementStatus.PENDING_APPROVAL: {DisbursementStatus.APPROVED, DisbursementStatus.CANCELLED},
        DisbursementStatus.APPROVED: {DisbursementStatus.SUBMISSION_PENDING, DisbursementStatus.SUBMITTED, DisbursementStatus.CANCELLED},
        DisbursementStatus.SUBMISSION_PENDING: {DisbursementStatus.SUBMITTED, DisbursementStatus.FAILED},
        DisbursementStatus.SUBMITTED: {DisbursementStatus.PROCESSING, DisbursementStatus.SUCCEEDED, DisbursementStatus.FAILED},
        DisbursementStatus.PROCESSING: {DisbursementStatus.SUCCEEDED, DisbursementStatus.FAILED, DisbursementStatus.HELD, DisbursementStatus.UNCLAIMED},
        DisbursementStatus.HELD: {DisbursementStatus.PROCESSING, DisbursementStatus.SUCCEEDED, DisbursementStatus.FAILED},
        DisbursementStatus.UNCLAIMED: {DisbursementStatus.RETURNED, DisbursementStatus.CANCELLED, DisbursementStatus.SUCCEEDED},
        DisbursementStatus.SUCCEEDED: {DisbursementStatus.REVERSED},
        DisbursementStatus.FAILED: set(),
        DisbursementStatus.RETURNED: set(),
        DisbursementStatus.CANCELLED: set(),
        DisbursementStatus.REVERSED: set(),
    }
    if target not in allowed.get(current, set()):
        raise ValidationException(f"Invalid disbursement transition from '{current.value}' to '{target.value}'.")
    return True


def test_valid_disbursement_lifecycle_transitions():
    """Verifies standard forward lifecycle transitions."""
    assert validate_disbursement_transition(DisbursementStatus.REQUESTED, DisbursementStatus.PENDING_APPROVAL)
    assert validate_disbursement_transition(DisbursementStatus.PENDING_APPROVAL, DisbursementStatus.APPROVED)
    assert validate_disbursement_transition(DisbursementStatus.APPROVED, DisbursementStatus.SUBMITTED)
    assert validate_disbursement_transition(DisbursementStatus.SUBMITTED, DisbursementStatus.PROCESSING)
    assert validate_disbursement_transition(DisbursementStatus.PROCESSING, DisbursementStatus.SUCCEEDED)


def test_invalid_disbursement_lifecycle_transitions():
    """Verifies that skipping approval or modifying terminal states raises ValidationException."""
    # Cannot jump from REQUESTED directly to SUCCEEDED
    with pytest.raises(ValidationException):
        validate_disbursement_transition(DisbursementStatus.REQUESTED, DisbursementStatus.SUCCEEDED)

    # Cannot transition out of SUCCEEDED except to REVERSED
    with pytest.raises(ValidationException):
        validate_disbursement_transition(DisbursementStatus.SUCCEEDED, DisbursementStatus.PENDING_APPROVAL)

    # Terminal FAILED cannot transition
    with pytest.raises(ValidationException):
        validate_disbursement_transition(DisbursementStatus.FAILED, DisbursementStatus.APPROVED)
