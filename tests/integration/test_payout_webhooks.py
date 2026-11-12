"""
test_payout_webhooks.py - Outbound PayPal Payout Webhook Ingestion Tests
========================================================================
Tests asynchronous webhook event processing:
1. PAYMENT.PAYOUTS-ITEM.SUCCEEDED -> marks disbursement SUCCEEDED and posts ledger execution.
2. PAYMENT.PAYOUTS-ITEM.FAILED -> marks disbursement FAILED and posts reversal.
3. Webhook event deduplication.
"""

import pytest
import uuid
import json
from datetime import datetime, timezone
from backend.app.core.constants import (
    DisbursementStatus, WebhookProcessingStatus, WebhookSignatureStatus
)
from backend.app.integrations.payments.exceptions import WebhookVerificationFailedException
from backend.app.db.models.financial import Disbursement, WebhookEvent
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.ledger_service import LedgerService
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def webhook_disb_setup(db_session):
    """Sets up a disbursement in PROCESSING state."""
    disb = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id="ms-wh-1",
        organization_id="org-wh-1",
        provider="PAYPAL_PAYOUTS",
        provider_item_id="MOCK-PAYOUT-ITEM-777",
        amount_minor_units=50000,
        currency="USD",
        status=DisbursementStatus.PROCESSING
    )
    db_session.add(disb)
    db_session.commit()

    # Pre-post commitment
    LedgerService.record_payout_commitment(db_session, disb)
    db_session.commit()

    return disb


def test_payout_item_succeeded_webhook(db_session, webhook_disb_setup):
    """Processes PAYMENT.PAYOUTS-ITEM.SUCCEEDED webhook and posts ledger execution."""
    disb = webhook_disb_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    resource_payload = {
        "payout_item_id": "MOCK-PAYOUT-ITEM-777",
        "payout_item": {
            "sender_item_id": disb.id,
            "amount": {"value": "500.00", "currency": "USD"}
        },
        "payout_item_fee": {"value": "0.25", "currency": "USD"},
        "transaction_status": "SUCCESS"
    }

    res = srv.process_payout_webhook_event(
        db=db_session,
        event_type="PAYMENT.PAYOUTS-ITEM.SUCCEEDED",
        resource=resource_payload
    )

    assert res["status"] == "SUCCESS"
    assert res["disbursement_id"] == disb.id

    db_session.refresh(disb)
    assert disb.status == DisbursementStatus.SUCCEEDED
    assert disb.disbursed_at is not None
    assert disb.provider_fee_minor_units == 25

    # Check ledger trial balance
    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True


def test_payout_item_failed_webhook(db_session, webhook_disb_setup):
    """Processes PAYMENT.PAYOUTS-ITEM.FAILED webhook and triggers ledger reversal."""
    disb = webhook_disb_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    resource_payload = {
        "payout_item_id": "MOCK-PAYOUT-ITEM-777",
        "payout_item": {
            "sender_item_id": disb.id
        },
        "errors": {"name": "RECEIVER_UNREGISTERED", "message": "Receiver account does not exist"}
    }

    res = srv.process_payout_webhook_event(
        db=db_session,
        event_type="PAYMENT.PAYOUTS-ITEM.FAILED",
        resource=resource_payload
    )

    assert res["status"] == "FAILED_RECORDED"

    db_session.refresh(disb)
    assert disb.status == DisbursementStatus.FAILED
    assert "RECEIVER_UNREGISTERED" in disb.failure_reason

    # Verify trial balance is still balanced after reversal
    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True


def test_payout_webhook_rejects_unverified_signature(db_session, webhook_disb_setup):
    """Verifies that unverified payout webhooks are rejected and do not modify financial status."""
    disb = webhook_disb_setup
    mock_provider = MockPayoutProvider()
    mock_provider.simulate_webhook_sig_failure = True
    srv = DisbursementService(provider=mock_provider)

    event_id = f"WH-UNVERIFIED-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": event_id,
        "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED",
        "resource_type": "payouts_item",
        "resource": {
            "payout_item_id": disb.provider_item_id,
            "payout_item": {
                "sender_item_id": disb.id,
                "amount": {"value": "500.00", "currency": "USD"}
            },
            "payout_item_fee": {"value": "0.25", "currency": "USD"},
            "transaction_status": "SUCCESS"
        }
    }
    raw_body = json.dumps(payload).encode("utf-8")
    headers = {"paypal-transmission-sig": "INVALID_SIGNATURE"}

    with pytest.raises(WebhookVerificationFailedException):
        srv.process_paypal_payout_webhook(db=db_session, headers=headers, raw_body=raw_body)

    # Verify disbursement was NOT modified
    db_session.refresh(disb)
    assert disb.status == DisbursementStatus.PROCESSING

    # Verify WebhookEvent recorded failure
    event_rec = db_session.query(WebhookEvent).filter_by(event_id=event_id).first()
    assert event_rec is not None
    assert event_rec.signature_verification_status == WebhookSignatureStatus.FAILED
    assert event_rec.processing_status == WebhookProcessingStatus.FAILED


def test_payout_webhook_accepts_verified_signature(db_session, webhook_disb_setup):
    """Verifies that verified payout webhooks succeed and update disbursement status."""
    disb = webhook_disb_setup
    mock_provider = MockPayoutProvider()
    srv = DisbursementService(provider=mock_provider)

    event_id = f"WH-VERIFIED-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": event_id,
        "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED",
        "resource_type": "payouts_item",
        "resource": {
            "payout_item_id": disb.provider_item_id,
            "payout_item": {
                "sender_item_id": disb.id,
                "amount": {"value": "500.00", "currency": "USD"}
            },
            "payout_item_fee": {"value": "0.25", "currency": "USD"},
            "transaction_status": "SUCCESS"
        }
    }
    raw_body = json.dumps(payload).encode("utf-8")
    headers = {
        "paypal-auth-algo": "SHA256withRSA",
        "paypal-cert-url": "https://api.sandbox.paypal.com/v1/cert",
        "paypal-transmission-id": "tx-5678",
        "paypal-transmission-sig": "VALID_SIGNATURE_BYTES_B64",
        "paypal-transmission-time": "2026-11-12T12:00:00Z"
    }

    res = srv.process_paypal_payout_webhook(db=db_session, headers=headers, raw_body=raw_body)
    assert res["status"] == "SUCCESS"

    # Verify disbursement was updated
    db_session.refresh(disb)
    assert disb.status == DisbursementStatus.SUCCEEDED
    assert disb.disbursed_at is not None

    # Verify WebhookEvent recorded verified success
    event_rec = db_session.query(WebhookEvent).filter_by(event_id=event_id).first()
    assert event_rec is not None
    assert event_rec.signature_verification_status == WebhookSignatureStatus.VERIFIED
    assert event_rec.processing_status == WebhookProcessingStatus.PROCESSED




