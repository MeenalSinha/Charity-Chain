"""
test_complete_donor_portfolio_workflow.py - End-to-End Donor Portfolio Workflow
===============================================================================
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
from backend.app.services.analytics.analytics_export_service import AnalyticsExportService


def test_full_donor_portfolio_lifecycle(db_session):
    """
    Workflow B — Donor Portfolio:
    1. Donor logs in.
    2. Opens personal portfolio.
    3. Views multi-currency giving overview.
    4. Tracks contribution journey into project, milestone, and verified certificate.
    5. Generates annual charitable giving report and downloads statement.
    """
    donor = User(
        id=str(uuid.uuid4()),
        email="generous_donor@example.org",
        full_name="Sarah Philanthropist",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(id=str(uuid.uuid4()), name="Rainforest Trust", creator_id=donor.id)
    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Canopy Protection",
        description="Preserving tropical canopy",
        target_amount_minor_units=1000000,
        currency="USD",
        status=ProjectStatus.ACTIVE
    )
    m = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Ranger Patrols",
        description="Fund wildlife rangers",
        target_amount_minor_units=500000,
        currency="USD",
        status=MilestoneStatus.COMPLETED
    )
    don = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=proj.id,
        milestone_id=m.id,
        amount_minor_units=25000,  # $250.00
        currency="USD",
        status=DonationStatus.PAID
    )
    cert = ImpactCertificate(
        id=str(uuid.uuid4()),
        certificate_number="CERT-CANOPY-007",
        certificate_type=CertificateType.CONTRIBUTION_IMPACT.value,
        status=CertificateStatus.ISSUED.value,
        recipient_name="Canopy Donor",
        organization_id=org.id,
        project_id=proj.id,
        milestone_id=m.id,
        donation_id=don.id,
        verification_id="token_canopy_7",
        signature="sig_valid_canopy",
        canonical_payload="{}",
        payload_hash="mock_hash_canopy"
    )

    db_session.add_all([donor, org, proj, m, don, cert])
    db_session.commit()

    # Step 2 & 3: Overview
    overview = DonorAnalyticsService.get_donor_overview(db_session, donor.id)
    assert overview["total_captured_cents"] == 25000
    assert overview["total_captured_formatted"] == "$250.00"
    assert overview["certificates_count"] == 1

    # Step 4: Personal Impact Journey
    journeys = DonorAnalyticsService.get_personal_impact_journey(db_session, donor.id)
    assert len(journeys) == 1
    assert journeys[0]["project"]["title"] == "Canopy Protection"
    assert journeys[0]["certificate"]["has_certificate"] is True
    assert journeys[0]["certificate"]["certificate_code"] == "CERT-CANOPY-007"

    # Step 5: Annual Statement & Export
    statement = DonorAnalyticsService.generate_annual_giving_report(db_session, donor.id, don.created_at.year)
    assert statement["total_contributions_cents"] == 25000
    assert len(statement["contributions"]) == 1

    csv_data = AnalyticsExportService.export_donor_contributions_csv(db_session, donor.id)
    assert "Canopy Protection" in csv_data
    assert "250.00" in csv_data
