"""
test_verification_jobs.py - Integration Tests for Verification Jobs & Persistence
=================================================================================
Tests verification job submission, idempotency, multi-indicator report persistence,
individual check record storage, and cryptographic audit hash chaining.
"""

import uuid
import pytest
from sqlalchemy import select

from backend.app.core.constants import PlatformRole, OrganizationRole, OrganizationStatus
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import (
    EvidenceFile, VerificationJob, VerificationReport,
    VerificationCheckRecord, EvidenceAuditEvent
)
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.verification_service import VerificationService
from backend.app.api.schemas import StartVerificationRequest


@pytest.fixture
def v_job_setup(db_session, sample_jpeg_bytes):
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"ngo_{uuid.uuid4().hex[:6]}@aid.org",
        hashed_password="pw",
        full_name="NGO Leader",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Relief NGO",
        legal_name="Global Relief NGO Inc",
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
        title="Solar Water Borehole",
        description="Community water supply project.",
        funding_target_minor=1000000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Pump Installation",
        description="Install water pump.",
        funding_target_minor=500000
    )
    db_session.add(milestone)
    db_session.commit()

    # Upload evidence file
    upload_res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone.id,
        files=[(sample_jpeg_bytes, "pump.jpg", "image/jpeg")],
        uploader_id=ngo_user.id
    )
    evidence_id = upload_res.items[0].id

    return {
        "ngo_user": ngo_user,
        "evidence_id": evidence_id,
        "milestone": milestone,
        "project": project
    }


def test_submit_verification_job_and_complete_report(db_session, v_job_setup):
    """Verify job is created, executed, and report persisted with check records."""
    setup = v_job_setup
    ev_id = setup["evidence_id"]
    user = setup["ngo_user"]

    job = VerificationService.submit_verification_job(
        db=db_session,
        evidence_id=ev_id,
        user=user
    )

    assert job.status in ["COMPLETED", "PROCESSING", "PENDING"]
    assert job.evidence_file_id == ev_id
    assert job.report_id is not None

    # Verify report in database
    report = db_session.execute(
        select(VerificationReport).where(VerificationReport.id == job.report_id)
    ).scalar_one_or_none()
    assert report is not None
    assert report.evidence_file_id == ev_id
    assert report.overall_assessment in ["CONSISTENT", "NEEDS_FURTHER_REVIEW", "INCONSISTENT"]
    assert len(report.checks) >= 3


def test_verification_job_idempotency(db_session, v_job_setup):
    """Verify duplicate submission with identical idempotency key returns existing job."""
    setup = v_job_setup
    ev_id = setup["evidence_id"]
    user = setup["ngo_user"]
    idem_key = f"idem_{uuid.uuid4().hex}"

    req = StartVerificationRequest(idempotency_key=idem_key)
    job1 = VerificationService.submit_verification_job(db=db_session, evidence_id=ev_id, user=user, req=req)
    job2 = VerificationService.submit_verification_job(db=db_session, evidence_id=ev_id, user=user, req=req)

    assert job1.id == job2.id
    assert job1.report_id == job2.report_id


def test_verification_audit_event_logged(db_session, v_job_setup):
    """Verify AI_VERIFIED audit event is appended to cryptographic hash chain."""
    setup = v_job_setup
    ev_id = setup["evidence_id"]
    user = setup["ngo_user"]

    VerificationService.submit_verification_job(db=db_session, evidence_id=ev_id, user=user)

    trail = EvidenceService.get_audit_trail(db_session, ev_id)
    assert trail.is_chain_intact is True
    event_types = [e.event_type for e in trail.events]
    assert "AI_VERIFIED" in event_types

