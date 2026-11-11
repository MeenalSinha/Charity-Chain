"""
test_payment_webhooks.py - PayPal Webhook Processing & Deduplication Integration Tests
======================================================================================
Tests cryptographic signature verification, event deduplication, and automatic capture/reconciliation.
"""

import pytest
import json
import uuid
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.core.constants import (
    ProjectStatus, MilestoneStatus, OrganizationStatus, DonationStatus,
    WebhookProcessingStatus
)
from backend.app.db.models.financial import WebhookEvent, Donation, PaymentOrder
from backend.app.services.donation_service import DonationService
from backend.app.integrations.payments.exceptions import WebhookVerificationFailedException
from tests.mocks.paypal_mock import MockPayPalProvider


@pytest.fixture
def webhook_project_setup(db_session):
    """Sets up project for webhook processing."""
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Oceans Initiative",
        legal_name="Clean Oceans Initiative Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Coastal Plastic Cleanup",
        description="Removing microplastics and ghost nets from marine reserves.",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Beach Cleaning Drive",
        description="First deployment of shoreline sweep teams.",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add(milestone)
    db_session.commit()

    return org, project, milestone


def test_webhook_deduplication(db_session, webhook_project_setup):
    """Verifies that receiving the identical webhook event twice acknowledges without duplicate side effects."""
    org, project, milestone = webhook_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    event_payload = {
        "id": "WH-EVENT-999-DEDUP",
        "event_type": "CHECKOUT.ORDER.APPROVED",
        "resource_type": "checkout-order",
        "resource": {"id": "NON-EXISTENT-ORDER"}
    }
    raw_body = json.dumps(event_payload).encode("utf-8")
    headers = {
        "PAYPAL-AUTH-ALGO": "SHA256withRSA",
        "PAYPAL-TRANSMISSION-ID": "TRANS-1",
        "PAYPAL-CERT-URL": "https://api.sandbox.paypal.com/cert",
        "PAYPAL-TRANSMISSION-SIG": "VALID_SIGNATURE_XYZ",
        "PAYPAL-TRANSMISSION-TIME": "2026-11-11T12:00:00Z"
    }

    # First delivery
    res1 = srv.process_paypal_webhook(db_session, headers=headers, raw_body=raw_body)
    assert res1["status"] == "SUCCESS"

    # Second delivery with identical event ID
    res2 = srv.process_paypal_webhook(db_session, headers=headers, raw_body=raw_body)
    assert res2["status"] == "DUPLICATE"


def test_webhook_invalid_signature_rejected(db_session, webhook_project_setup):
    """Verifies that unverified or forged webhook signatures raise WebhookVerificationFailedException."""
    org, project, milestone = webhook_project_setup
    mock_provider = MockPayPalProvider()
    mock_provider.simulate_invalid_webhook = True
    srv = DonationService(provider=mock_provider)

    event_payload = {
        "id": "WH-EVENT-FORGED",
        "event_type": "PAYMENT.CAPTURE.COMPLETED",
        "resource": {"id": "CAP-FORGED"}
    }
    raw_body = json.dumps(event_payload).encode("utf-8")
    headers = {"PAYPAL-TRANSMISSION-SIG": "FORGED_SIGNATURE"}

    with pytest.raises(WebhookVerificationFailedException):
        srv.process_paypal_webhook(db_session, headers=headers, raw_body=raw_body)


def test_webhook_auto_captures_approved_order(db_session, webhook_project_setup):
    """
    Verifies that CHECKOUT.ORDER.APPROVED webhook automatically triggers payment capture
    even if the donor closed their browser before returning.
    """
    org, project, milestone = webhook_project_setup
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    # 1. Create donation intent
    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=8000,
        currency="USD"
    )

    # 2. Simulate webhook CHECKOUT.ORDER.APPROVED from PayPal
    event_payload = {
        "id": f"WH-EVENT-{uuid.uuid4()}",
        "event_type": "CHECKOUT.ORDER.APPROVED",
        "resource": {"id": intent["provider_order_id"]}
    }
    raw_body = json.dumps(event_payload).encode("utf-8")
    headers = {
        "PAYPAL-AUTH-ALGO": "SHA256withRSA",
        "PAYPAL-TRANSMISSION-ID": "TRANS-2",
        "PAYPAL-CERT-URL": "https://cert",
        "PAYPAL-TRANSMISSION-SIG": "VALID_SIG",
        "PAYPAL-TRANSMISSION-TIME": "2026-11-11T12:05:00Z"
    }

    res = srv.process_paypal_webhook(db_session, headers=headers, raw_body=raw_body)
    assert res["status"] == "SUCCESS"

    # 3. Verify donation was automatically captured and marked PAID
    don = db_session.query(Donation).filter(Donation.id == intent["donation_id"]).first()
    assert don.status == DonationStatus.PAID

    db_session.refresh(milestone)
    assert milestone.raised_amount_minor_units == 8000
