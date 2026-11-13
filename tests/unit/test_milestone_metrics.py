"""
test_milestone_metrics.py - Milestone Aggregation & Progress Unit Tests
======================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization, User
from backend.app.core.constants import MilestoneStatus, ProjectStatus, PlatformRole, AccountStatus
from backend.app.services.analytics.ngo_analytics_service import NGOAnalyticsService


def test_milestone_and_project_completion_rates(db_session):
    """Verifies that milestone progress and project completion rates calculate accurately."""
    user = User(
        id=str(uuid.uuid4()),
        email="milestone_test@example.org",
        full_name="Milestone User",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Education NGO",
        creator_id=user.id
    )
    # Project 1: Target $10,000, 2 milestones
    proj1 = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Books For Kids",
        description="Textbook distribution",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=1000000,
        currency="USD",
        status=ProjectStatus.ACTIVE
    )
    m1 = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj1.id,
        sequence_number=1,
        title="Purchase Books",
        description="Acquire books",
        target_amount_minor_units=500000,
        currency="USD",
        status=MilestoneStatus.COMPLETED  # Completed
    )
    m2 = Milestone(
        id=str(uuid.uuid4()),
        project_id=proj1.id,
        sequence_number=2,
        title="Distribute Books",
        description="Deliver to schools",
        target_amount_minor_units=500000,
        currency="USD",
        status=MilestoneStatus.SUBMITTED  # In progress
    )
    db_session.add_all([user, org, proj1, m1, m2])
    db_session.commit()

    perf = NGOAnalyticsService.get_project_performance(db_session, org.id)
    assert len(perf) == 1
    p_stat = perf[0]

    assert p_stat["milestones_total"] == 2
    assert p_stat["milestones_completed"] == 1
    assert p_stat["completion_percent"] == 50.0
    assert p_stat["funding_progress_percent"] == 100.0
