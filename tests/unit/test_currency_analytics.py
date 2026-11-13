"""
test_currency_analytics.py - Multi-Currency & Minor Units Arithmetic Tests
==========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.financial import Donation
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.core.constants import DonationStatus, PlatformRole, AccountStatus
from backend.app.services.analytics.donor_analytics_service import DonorAnalyticsService


def test_donor_multi_currency_segregation(db_session):
    """Verifies that donations in different currencies are segregated and not blindly added together."""
    donor = User(
        id=str(uuid.uuid4()),
        email="multicurrency_donor@example.org",
        full_name="Multi Currency Donor",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Global NGO",
        creator_id=donor.id
    )
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Clean Water Project",
        description="Global Water Initiative",
        target_amount_minor_units=5000000,
        currency="USD"
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Phase 1 Wells",
        description="Drilling initial wells",
        target_amount_minor_units=2500000,
        currency="USD"
    )
    db_session.add_all([donor, org, proj, m])
    db_session.commit()

    # 1 donation in USD (100.00 USD = 10000 cents)
    d_usd = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=proj.id,
        milestone_id=m.id,
        amount_minor_units=10000,
        currency="USD",
        status=DonationStatus.PAID
    )
    # 1 donation in EUR (50.00 EUR = 5000 cents)
    d_eur = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=proj.id,
        milestone_id=m.id,
        amount_minor_units=5000,
        currency="EUR",
        status=DonationStatus.PAID
    )
    db_session.add_all([d_usd, d_eur])
    db_session.commit()

    overview = DonorAnalyticsService.get_donor_overview(db_session, donor.id)

    # Check that currencies are tracked in separate buckets
    currencies = overview["currencies"]
    assert "USD" in currencies
    assert "EUR" in currencies
    assert currencies["USD"]["captured_cents"] == 10000
    assert currencies["EUR"]["captured_cents"] == 5000
