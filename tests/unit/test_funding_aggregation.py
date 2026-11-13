"""
test_funding_aggregation.py - Funding Calculation & Refund Deduction Unit Tests
===============================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentCapture, PaymentRefund
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.core.constants import (
    DonationStatus, PaymentCaptureStatus, PaymentRefundStatus,
    PlatformRole, AccountStatus
)
from backend.app.services.analytics.platform_analytics_service import PlatformAnalyticsService


def test_captured_donations_exclude_pending_and_deduct_refunds(db_session):
    """Verifies that pending captures and refunds are accurately calculated."""
    user = User(
        id=str(uuid.uuid4()),
        email="funding_test@example.org",
        full_name="Funding User",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Funding NGO",
        creator_id=user.id
    )
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Solar Initiative",
        description="Renewable Energy",
        target_amount_minor_units=1000000,
        currency="USD"
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Solar Panels",
        description="Installing panels",
        target_amount_minor_units=500000,
        currency="USD"
    )
    db_session.add_all([user, org, proj, m])
    db_session.commit()

    # 1. Successful capture: $200.00 (20000 cents)
    c1 = PaymentCapture(
        id=str(uuid.uuid4()),
        amount_minor_units=20000,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED
    )
    # 2. Pending capture: $150.00 (15000 cents) - MUST NOT BE COUNTED
    c2 = PaymentCapture(
        id=str(uuid.uuid4()),
        amount_minor_units=15000,
        currency="USD",
        status=PaymentCaptureStatus.PENDING
    )
    # 3. Failed/declined capture: $50.00 (5000 cents) - MUST NOT BE COUNTED
    c3 = PaymentCapture(
        id=str(uuid.uuid4()),
        amount_minor_units=5000,
        currency="USD",
        status=PaymentCaptureStatus.DECLINED
    )
    # 4. Refund of $50.00 (5000 cents) against c1
    r1 = PaymentRefund(
        id=str(uuid.uuid4()),
        capture_id=c1.id,
        amount_minor_units=5000,
        currency="USD",
        status=PaymentRefundStatus.COMPLETED
    )
    db_session.add_all([c1, c2, c3, r1])
    db_session.commit()

    overview = PlatformAnalyticsService.get_platform_overview(db_session)
    fin = overview["financial"]

    assert fin["captured_cents"] == 20000
    assert fin["refunded_cents"] == 5000
    assert fin["net_funding_cents"] == 15000
    assert fin["captured_formatted"] == "$200.00"
    assert fin["net_funding_formatted"] == "$150.00"
