"""
test_payment_state_machine.py - Payment & Donation State Machine Unit Tests
============================================================================
Enforces valid and invalid lifecycle transitions for donations, orders, captures, and refunds.
"""

import pytest
from backend.app.core.constants import (
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus, PaymentRefundStatus
)
from backend.app.core.state_machines import StateMachineValidator
from backend.app.core.exceptions import InvalidStateTransitionError


def test_donation_valid_transitions():
    """Tests compliant progressions through the donation lifecycle."""
    # Standard happy path
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.CREATED, DonationStatus.PAYMENT_PENDING
    )
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.PAYMENT_PENDING, DonationStatus.PAID
    )

    # Refund paths
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.PAID, DonationStatus.PARTIALLY_REFUNDED
    )
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.PARTIALLY_REFUNDED, DonationStatus.REFUNDED
    )
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.PAID, DonationStatus.REFUNDED
    )

    # Cancellation paths
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.CREATED, DonationStatus.CANCELLED
    )
    assert StateMachineValidator.validate_donation_transition(
        DonationStatus.PAYMENT_PENDING, DonationStatus.CANCELLED
    )


def test_donation_invalid_transitions():
    """Ensures forbidden transitions are strictly rejected."""
    # Cannot jump directly from CREATED to PAID without pending authorization
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_donation_transition(
            DonationStatus.CREATED, DonationStatus.PAID
        )

    # Cannot transition from CANCELLED to PAID
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_donation_transition(
            DonationStatus.CANCELLED, DonationStatus.PAID
        )

    # Cannot transition from REFUNDED to PAID
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_donation_transition(
            DonationStatus.REFUNDED, DonationStatus.PAID
        )

    # Cannot transition from FAILED to PAID
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_donation_transition(
            DonationStatus.FAILED, DonationStatus.PAID
        )


def test_payment_order_valid_and_invalid_transitions():
    """Tests payment order transitions."""
    assert StateMachineValidator.validate_payment_order_transition(
        PaymentOrderStatus.CREATED, PaymentOrderStatus.APPROVED
    )
    assert StateMachineValidator.validate_payment_order_transition(
        PaymentOrderStatus.APPROVED, PaymentOrderStatus.COMPLETED
    )
    assert StateMachineValidator.validate_payment_order_transition(
        PaymentOrderStatus.CREATED, PaymentOrderStatus.VOIDED
    )

    # Completed is terminal
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_payment_order_transition(
            PaymentOrderStatus.COMPLETED, PaymentOrderStatus.CREATED
        )


def test_payment_capture_valid_and_invalid_transitions():
    """Tests payment capture transitions."""
    assert StateMachineValidator.validate_payment_capture_transition(
        PaymentCaptureStatus.PENDING, PaymentCaptureStatus.COMPLETED
    )
    assert StateMachineValidator.validate_payment_capture_transition(
        PaymentCaptureStatus.COMPLETED, PaymentCaptureStatus.PARTIALLY_REFUNDED
    )
    assert StateMachineValidator.validate_payment_capture_transition(
        PaymentCaptureStatus.PARTIALLY_REFUNDED, PaymentCaptureStatus.REFUNDED
    )

    # Refunded is terminal
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_payment_capture_transition(
            PaymentCaptureStatus.REFUNDED, PaymentCaptureStatus.COMPLETED
        )

