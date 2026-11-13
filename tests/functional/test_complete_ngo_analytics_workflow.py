"""
test_complete_ngo_analytics_workflow.py - End-to-End NGO Analytics Workflow
===========================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import Donation, PaymentCapture, Disbursement
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, AccountStatus, ProjectStatus,
    MilestoneStatus, DonationStatus, PaymentCaptureStatus, DisbursementStatus
)
from backend.app.services.analytics.ngo_analytics_service import NGOAnalyticsService
from backend.app.services.analytics.analytics_export_service import AnalyticsExportService


def test_full_ngo_analytics_lifecycle(db_session):
    """
    Workflow A — NGO Analytics:
    1. NGO onboarded.
    2. Opens analytics overview.
    3. Examines funding and milestone progress.
    4. Generates an organization impact report.
    5. Exports project performance CSV.
    """
    admin = User(
        id=str(uuid.uuid4()),
        email="director@globalhealth.org",
        full_name="Dr. Jane Director",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(id=str(uuid.uuid4()), name="Global Health Alliance", creator_id=admin.id)
    mem = OrganizationMembership(id=str(uuid.uuid4()), user_id=admin.id, organization_id=org.id, role=OrganizationRole.OWNER, is_active=True)

    proj = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Mobile Vaccination Clinics",
        description="Deploying mobile clinics",
        target_amount_minor_units=2000000,
        raised_amount_minor_units=1500000,
        currency="USD",
        status=ProjectStatus.ACTIVE
    )
    m1 = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=1,
        title="Procure Vaccines",
        description="Cold chain procurement",
        target_amount_minor_units=1000000,
        currency="USD",
        status=MilestoneStatus.COMPLETED
    )
    m2 = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj.id,
        sequence_number=2,
        title="Deploy Fleet",
        description="Fleet deployment",
        target_amount_minor_units=500000,
        currency="USD",
        status=MilestoneStatus.ACTIVE
    )
    don = Donation(
        id=str(uuid.uuid4()),
        donor_id=admin.id,
        project_id=proj.id,
        milestone_id=m1.id,
        amount_minor_units=1500000,
        currency="USD",
        status=DonationStatus.PAID
    )
    disb = Disbursement(
        id=str(uuid.uuid4()),
        milestone_id=m1.id,
        organization_id=org.id,
        amount_minor_units=1000000,
        currency="USD",
        status=DisbursementStatus.SUCCEEDED
    )

    db_session.add_all([admin, org, mem, proj, m1, m2, don, disb])
    db_session.commit()

    # Step 2: Query Overview
    overview = NGOAnalyticsService.get_ngo_overview(db_session, org.id)
    assert overview["projects"]["total"] == 1
    assert overview["projects"]["active"] == 1
    assert overview["financial"]["total_captured_cents"] == 1500000
    assert overview["financial"]["confirmed_disbursements_cents"] == 1000000
    assert overview["financial"]["available_funds_cents"] == 500000

    # Step 3: Funding & Project Performance
    funding = NGOAnalyticsService.get_funding_analytics(db_session, org.id)
    assert funding["total_raised_cents"] == 1500000
    assert len(funding["funding_by_project"]) == 1

    projects_perf = NGOAnalyticsService.get_project_performance(db_session, org.id)
    assert projects_perf[0]["milestones_total"] == 2
    assert projects_perf[0]["milestones_completed"] == 1

    # Step 4: Generate Report
    report = NGOAnalyticsService.generate_impact_report(db_session, org.id)
    assert report["organization_name"] == "Global Health Alliance"
    assert "disclaimer" in report

    # Step 5: Export CSV
    csv_text = AnalyticsExportService.export_ngo_projects_csv(db_session, org.id)
    assert "Mobile Vaccination Clinics" in csv_text
    assert "15000.00" in csv_text
