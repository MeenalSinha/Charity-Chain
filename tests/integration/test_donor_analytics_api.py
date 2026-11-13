"""
test_donor_analytics_api.py - Integration Tests for Donor Portfolio & Impact Journeys
=====================================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import Donation
from backend.app.db.models.reviews_impact import ImpactCertificate
from backend.app.core.constants import (
    PlatformRole, AccountStatus, ProjectStatus, MilestoneStatus, DonationStatus,
    CertificateType, CertificateStatus
)
from backend.app.services.analytics.donor_analytics_service import DonorAnalyticsService


@pytest.fixture
def donor_portfolio_env(db_session):
    donor = User(
        id=str(uuid.uuid4()),
        email=f"donor_{uuid.uuid4().hex[:6]}@example.org",
        full_name="Impact Donor",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Oceans Foundation",
        creator_id=donor.id
    )
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Pacific Ocean Plastic Cleanup",
        description="Ocean conservation",
        target_amount_minor_units=2000000,
        currency="USD",
        status=ProjectStatus.ACTIVE
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Deploy Ocean Barrier",
        description="Deploy barrier system",
        target_amount_minor_units=1000000,
        currency="USD",
        status=MilestoneStatus.COMPLETED
    )
    don = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=proj.id,
        milestone_id=m.id,
        amount_minor_units=50000,  # $500.00
        currency="USD",
        status=DonationStatus.PAID
    )
    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        certificate_number="CERT-PACIFIC-001",
        certificate_type=CertificateType.CONTRIBUTION_IMPACT.value,
        status=CertificateStatus.ISSUED.value,
        recipient_name="Pacific Donor",
        organization_id=org.id,
        project_id=proj.id,
        milestone_id=m.id,
        donation_id=don.id,
        verification_id="tok_test_pacific",
        signature="sig_valid_test",
        canonical_payload="{}",
        payload_hash="mock_hash"
    )
    db_session.add_all([donor, org, proj, m, don, cert])
    db_session.commit()
    return {"donor": donor, "proj": proj, "milestone": m, "donation": don, "cert": cert}


def test_donor_portfolio_and_impact_journey(db_session, donor_portfolio_env):
    """Verifies personal giving portfolio and end-to-end journey attribution."""
    donor_id = donor_portfolio_env["donor"].id

    overview = DonorAnalyticsService.get_donor_overview(db_session, donor_id)
    assert overview["total_captured_cents"] == 50000
    assert overview["total_captured_formatted"] == "$500.00"
    assert overview["supported_projects_count"] == 1
    assert overview["certificates_count"] == 1

    # Personal impact journey
    journeys = DonorAnalyticsService.get_personal_impact_journey(db_session, donor_id)
    assert len(journeys) == 1
    j = journeys[0]

    assert j["amount_formatted"] == "$500.00"
    assert j["project"]["title"] == "Pacific Ocean Plastic Cleanup"
    assert j["milestone"]["sequence_number"] == 1
    assert j["is_completed"] is True
    assert j["certificate"]["has_certificate"] is True
    assert j["certificate"]["certificate_code"] == "CERT-PACIFIC-001"


def test_annual_giving_report(db_session, donor_portfolio_env):
    """Verifies that the annual contribution statement contains tax disclaimer and line items."""
    donor_id = donor_portfolio_env["donor"].id
    current_year = donor_portfolio_env["donation"].created_at.year

    report = DonorAnalyticsService.generate_annual_giving_report(db_session, donor_id, current_year)
    assert report["tax_year"] == current_year
    assert report["total_contributions_cents"] == 50000
    assert report["total_contributions_formatted"] == "$500.00"
    assert len(report["contributions"]) == 1
    assert "tax advisor" in report["tax_disclaimer"]
