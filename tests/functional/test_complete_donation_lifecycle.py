"""
test_complete_donation_lifecycle.py - Complete End-to-End Charitable Giving & Refund Scenario
=============================================================================================
Exercises the complete 10-step lifecycle specified in Phase 4 Task 22:
Discovery -> Milestone Selection -> PayPal Order -> Checkout Approval -> Backend Capture ->
Webhook Reconciliation -> Funding Aggregation -> Donor Portfolio -> Partial Refund.
"""

import pytest
import uuid
import json
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture, PaymentRefund
from backend.app.core.constants import (
    PlatformRole, AccountStatus, OrganizationStatus,
    ProjectStatus, MilestoneStatus, ProjectVisibility, DonationStatus
)
from backend.app.services.donation_service import DonationService
from tests.mocks.paypal_mock import MockPayPalProvider


def test_full_donation_and_refund_lifecycle_e2e(db_session):
    """Executes the comprehensive Phase 4 end-to-end acceptance test."""
    mock_provider = MockPayPalProvider()
    srv = DonationService(provider=mock_provider)

    # 1. Provision Verified NGO
    org = Organization(
        id=str(uuid.uuid4()),
        name="Hope International",
        legal_name="Hope International Charitable Trust",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    # 2. Provision Active Public Project with 2 Milestones
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Safe Drinking Water Project",
        description="Installing water purification units across rural schools.",
        project_type="INFRASTRUCTURE",
        public_visibility=ProjectVisibility.PUBLIC,
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,  # $10,000.00
        raised_amount_minor_units=0
    )
    db_session.add(project)

    m1 = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Water Quality Assessment & Permitting",
        description="Hydrogeological surveying and state environmental agency permitting.",
        target_amount_minor_units=300000,  # $3,000.00
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    m2 = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=2,
        title="Filtration Unit Installation",
        description="Procuring and mounting reverse-osmosis filtration hardware.",
        target_amount_minor_units=700000,  # $7,000.00
        raised_amount_minor_units=0,
        status=MilestoneStatus.ACTIVE
    )
    db_session.add_all([m1, m2])

    # 3. Provision Authenticated Donor
    donor = User(
        id=str(uuid.uuid4()),
        email="generous.donor@example.com",
        full_name="Generous Donor",
        hashed_password="mock_argon2_hash",
        platform_role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(donor)
    db_session.commit()

    # 4. Donor creates donation intent: $100.00 (10,000 minor units) for Milestone #1
    intent = srv.create_donation_intent(
        db=db_session,
        project_id=project.id,
        milestone_id=m1.id,
        amount_minor_units=10000,
        currency="USD",
        donor_id=donor.id,
        donor_name="Generous Donor",
        donor_email=donor.email
    )
    assert intent["status"] == DonationStatus.PAYMENT_PENDING.value
    assert "token=" in intent["approval_url"]
    provider_order_id = intent["provider_order_id"]

    # 5. Donor approves payment at PayPal and returns; Backend executes payment capture
    capture_res = srv.capture_donation_payment(
        db=db_session,
        order_identifier=provider_order_id,
        user_id=donor.id
    )
    assert capture_res["status"] == "COMPLETED"
    assert capture_res["donation_status"] == DonationStatus.PAID.value

    # 6. Verify funding totals incremented
    db_session.refresh(project)
    db_session.refresh(m1)
    assert project.raised_amount_minor_units == 10000
    assert m1.raised_amount_minor_units == 10000

    # 7. Asynchronous webhook arrives from PayPal (CHECKOUT.ORDER.APPROVED / PAYMENT.CAPTURE.COMPLETED)
    webhook_event = {
        "id": f"WH-{uuid.uuid4()}",
        "event_type": "PAYMENT.CAPTURE.COMPLETED",
        "resource": {
            "id": capture_res["provider_capture_id"],
            "custom_id": intent["donation_id"]
        }
    }
    raw_body = json.dumps(webhook_event).encode("utf-8")
    headers = {
        "PAYPAL-AUTH-ALGO": "SHA256withRSA",
        "PAYPAL-TRANSMISSION-ID": f"TR-{uuid.uuid4()}",
        "PAYPAL-CERT-URL": "https://cert.paypal.com",
        "PAYPAL-TRANSMISSION-SIG": "VALID_SIGNATURE",
        "PAYPAL-TRANSMISSION-TIME": "2026-11-11T13:00:00Z"
    }
    wh_res = srv.process_paypal_webhook(db=db_session, headers=headers, raw_body=raw_body)
    assert wh_res["status"] == "SUCCESS"

    # Verify deduplication
    wh_dup = srv.process_paypal_webhook(db=db_session, headers=headers, raw_body=raw_body)
    assert wh_dup["status"] == "DUPLICATE"

    # 8. Donor inspects donation portfolio in dashboard
    donations, count = srv.list_donor_donations(db=db_session, donor_id=donor.id)
    assert count == 1
    assert donations[0].status == DonationStatus.PAID
    assert donations[0].amount_minor_units == 10000

    # 9. Donor requests a partial refund: $30.00 (3,000 minor units)
    refund_res = srv.process_refund(
        db=db_session,
        donation_id=intent["donation_id"],
        amount_minor_units=3000,
        reason="Incorrect contribution amount entered",
        actor_id=donor.id
    )
    assert refund_res["status"] == "COMPLETED"
    assert refund_res["donation_status"] == DonationStatus.PARTIALLY_REFUNDED.value
    assert refund_res["remaining_refundable_minor_units"] == 7000

    # 10. Re-verify net funding totals adjusted accurately
    db_session.refresh(project)
    db_session.refresh(m1)
    assert project.raised_amount_minor_units == 7000  # $100.00 - $30.00 = $70.00
    assert m1.raised_amount_minor_units == 7000

    # Summary verification
    funding_summary = srv.get_project_funding_summary(db=db_session, project_id=project.id)
    assert funding_summary["gross_received_minor_units"] == 10000
    assert funding_summary["refunded_minor_units"] == 3000
    assert funding_summary["net_received_minor_units"] == 7000
