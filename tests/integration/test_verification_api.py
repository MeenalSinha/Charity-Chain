"""
test_verification_api.py - Integration Tests for AI Verification REST Endpoints
===============================================================================
Tests API endpoints for starting verification jobs, checking job status,
retrieving latest reports, history pagination, and report details.
"""

import uuid
import pytest
from backend.app.core.constants import PlatformRole, OrganizationRole, OrganizationStatus
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.services.evidence_service import EvidenceService
from backend.app.api.v1.verification import (
    start_evidence_verification, get_verification_job_status,
    get_latest_verification_report, get_verification_history,
    get_verification_report_detail
)
from backend.app.api.schemas import StartVerificationRequest


@pytest.fixture
def api_test_setup(db_session, sample_jpeg_bytes):
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"api_ngo_{uuid.uuid4().hex[:6]}@aid.org",
        hashed_password="pw",
        full_name="API NGO Lead",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="API Test NGO",
        legal_name="API Test NGO Inc",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    mem = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_user.id,
        role=OrganizationRole.ADMIN,
        is_active=True
    )
    db_session.add(mem)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Solar Power Array",
        description="Renewable solar array.",
        funding_target_minor=1000000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Panel Installation",
        description="Install panels.",
        funding_target_minor=500000
    )
    db_session.add(milestone)
    db_session.commit()

    upload_res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone.id,
        files=[(sample_jpeg_bytes, "panel.jpg", "image/jpeg")],
        uploader_id=ngo_user.id
    )
    ev_id = upload_res.items[0].id

    return {"user": ngo_user, "evidence_id": ev_id}


def test_api_verification_lifecycle_endpoints(db_session, api_test_setup):
    """Test full API endpoint sequence: start -> status -> latest -> history -> detail."""
    user = api_test_setup["user"]
    ev_id = api_test_setup["evidence_id"]

    # 1. Start Verification
    req = StartVerificationRequest(idempotency_key="api_test_idem_1")
    start_resp = start_evidence_verification(
        evidence_id=ev_id,
        req=req,
        current_user=user,
        db=db_session
    )
    assert start_resp.success is True
    job_id = start_resp.data.id

    # 2. Get Job Status
    status_resp = get_verification_job_status(
        job_id=job_id,
        current_user=user,
        db=db_session
    )
    assert status_resp.success is True
    assert status_resp.data.id == job_id
    assert status_resp.data.status in ["COMPLETED", "PROCESSING", "PENDING"]
    report_id = status_resp.data.report_id
    assert report_id is not None

    # 3. Get Latest Report
    latest_resp = get_latest_verification_report(
        evidence_id=ev_id,
        current_user=user,
        db=db_session
    )
    assert latest_resp.success is True
    assert latest_resp.data.id == report_id
    assert latest_resp.data.evidence_file_id == ev_id

    # 4. Get History
    hist_resp = get_verification_history(
        evidence_id=ev_id,
        limit=10,
        offset=0,
        current_user=user,
        db=db_session
    )
    assert hist_resp.success is True
    assert hist_resp.data.total_reports >= 1
    assert hist_resp.data.reports[0].id == report_id

    # 5. Get Report Detail
    detail_resp = get_verification_report_detail(
        report_id=report_id,
        current_user=user,
        db=db_session
    )
    assert detail_resp.success is True
    assert detail_resp.data.id == report_id
    assert len(detail_resp.data.checks) > 0

