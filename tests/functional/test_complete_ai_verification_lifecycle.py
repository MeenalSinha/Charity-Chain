"""
test_complete_ai_verification_lifecycle.py - End-to-End Functional Test
========================================================================
Validates complete multi-indicator AI verification lifecycle from evidence upload,
integrity validation, job queueing, multi-check execution, report retrieval,
versioned analysis isolation, and cryptographic audit hash chaining.
"""

import uuid
import pytest
from backend.app.core.constants import PlatformRole, OrganizationRole, OrganizationStatus
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.verification_service import VerificationService
from backend.app.api.schemas import StartVerificationRequest


def test_complete_ai_verification_lifecycle(db_session, sample_jpeg_bytes, sample_modified_jpeg_bytes):
    """
    End-to-end verification lifecycle:
    1. Setup NGO, Organization, Project, Milestone
    2. Upload baseline photograph & milestone progress evidence
    3. Run AI verification job linking baseline evidence
    4. Verify multi-indicator report, checks, and overall assessment
    5. Upload replacement version v2 and verify v1 report remains immutable
    6. Verify audit hash chain preserves AI_VERIFIED events
    """
    # 1. Setup entities
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"director_{uuid.uuid4().hex[:6]}@aid.org",
        hashed_password="pw",
        full_name="Program Director",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Forestry Alliance",
        legal_name="Global Forestry Alliance Ltd",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    mem = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_user.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    db_session.add(mem)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Boreal Tree Restoration",
        description="Planting coniferous saplings in degraded forest land.",
        funding_target_minor=1500000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Sapling Planting",
        description="Plant 10,000 saplings.",
        funding_target_minor=750000
    )
    db_session.add(milestone)
    db_session.commit()

    # 2. Upload baseline photo & milestone evidence photo
    baseline_batch = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone.id,
        files=[(sample_jpeg_bytes, "cleared_land_baseline.jpg", "image/jpeg")],
        uploader_id=ngo_user.id
    )
    baseline_id = baseline_batch.items[0].id

    progress_batch = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone.id,
        files=[(sample_modified_jpeg_bytes, "saplings_planted.jpg", "image/jpeg")],
        uploader_id=ngo_user.id
    )
    progress_id = progress_batch.items[0].id

    # 3. Submit AI Verification job linking baseline evidence
    v_req = StartVerificationRequest(baseline_evidence_id=baseline_id)
    job = VerificationService.submit_verification_job(
        db=db_session,
        evidence_id=progress_id,
        user=ngo_user,
        req=v_req
    )

    assert job.status in ["COMPLETED", "PROCESSING", "PENDING"]
    assert job.report_id is not None

    # 4. Query Latest Report
    report = VerificationService.get_latest_report(db=db_session, evidence_id=progress_id, user=ngo_user)
    assert report.id == job.report_id
    assert report.evidence_file_id == progress_id
    assert report.overall_assessment in ["CONSISTENT", "NEEDS_FURTHER_REVIEW", "INCONSISTENT"]
    assert report.confidence_score is not None
    assert report.calibration_status == "UNVALIDATED"
    assert len(report.checks) >= 4

    # 5. Upload version 2 of progress evidence
    v2_bytes = b"\xff\xd8\xff\xe0JFIF revised v2 sapling photo"
    v2_ev = EvidenceService.upload_evidence_version(
        db=db_session,
        parent_evidence_id=progress_id,
        file_bytes=v2_bytes,
        filename="saplings_planted_v2.jpg",
        mime_type="image/jpeg",
        uploader_id=ngo_user.id,
        reason="Updated high-resolution aerial capture"
    )
    assert v2_ev.version_number == 2

    # Verify v1 report is still accessible and unchanged in history
    hist = VerificationService.get_verification_history(db=db_session, evidence_id=progress_id, user=ngo_user)
    assert hist.total_reports >= 1
    assert hist.reports[0].id == report.id
    assert hist.reports[0].evidence_version == 1

    # 6. Verify audit hash chain
    trail = EvidenceService.get_audit_trail(db_session, progress_id)
    assert trail.is_chain_intact is True
    event_types = [e.event_type for e in trail.events]
    assert "UPLOADED" in event_types
    assert "AI_VERIFIED" in event_types

