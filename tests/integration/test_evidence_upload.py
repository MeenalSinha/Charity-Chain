"""
test_evidence_upload.py - Integration Tests for Multi-File Evidence Uploads
===========================================================================
Verifies multi-file batch processing, permission checks, duplicate detection,
and audit record initialization.
"""

import uuid
import pytest
from sqlalchemy import select

from backend.app.core.constants import OrganizationRole, OrganizationStatus, PlatformRole, UserRole
from backend.app.core.exceptions import ConflictException, ValidationException, PermissionDeniedError, NotFoundException
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import EvidenceFile, EvidenceAuditEvent
from backend.app.services.evidence_service import EvidenceService


@pytest.fixture
def evidence_test_setup(db_session):
    """Sets up an NGO user, an organization, a project, and a milestone."""
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"ngo_{uuid.uuid4().hex[:6]}@relief.org",
        hashed_password="hashed_pw_test",
        full_name="NGO Project Lead",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    other_user = User(
        id=str(uuid.uuid4()),
        email=f"donor_{uuid.uuid4().hex[:6]}@gmail.com",
        hashed_password="hashed_pw_test",
        full_name="Public Donor",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(other_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Clean Water Initiative",
        legal_name="Clean Water Initiative NGO",
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
        title="Community Solar Well",
        description="Providing solar-powered water extraction for village irrigation.",
        funding_target_minor=1000000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Solar Array & Pump Installation",
        description="Procure solar panels, inverter, and submersible pump.",
        funding_target_minor=500000
    )
    db_session.add(milestone)
    db_session.commit()

    return {
        "ngo_user": ngo_user,
        "other_user": other_user,
        "org": org,
        "project": project,
        "milestone": milestone
    }


def test_successful_multi_file_evidence_upload(db_session, evidence_test_setup):
    """Verify NGO user can upload multiple evidence files in a single batch."""
    setup = evidence_test_setup
    m_id = setup["milestone"].id
    u_id = setup["ngo_user"].id

    file1 = (b"%PDF-1.4 Water quality report content", "water_quality.pdf", "application/pdf")
    file2 = (b"\xff\xd8\xff\xe0JFIF Solar panels photo", "solar_panels.jpg", "image/jpeg")

    batch_resp = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[file1, file2],
        uploader_id=u_id
    )

    assert batch_resp.total_uploaded == 2
    assert len(batch_resp.items) == 2
    assert batch_resp.items[0].milestone_id == m_id
    assert batch_resp.items[0].version_number == 1
    assert batch_resp.items[0].status == "UPLOADED"

    # Verify audit event creation
    for item in batch_resp.items:
        audits = db_session.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == item.id)
        ).scalars().all()
        assert len(audits) == 1
        assert audits[0].event_type == "UPLOADED"
        assert audits[0].previous_event_hash == "0" * 64
        assert len(audits[0].current_event_hash) == 64


def test_evidence_upload_detects_duplicates(db_session, evidence_test_setup):
    """Verify attempting to upload a file with identical SHA-256 hash triggers ConflictException."""
    setup = evidence_test_setup
    m_id = setup["milestone"].id
    u_id = setup["ngo_user"].id

    file1 = (b"%PDF-1.4 Duplicate candidate document", "doc.pdf", "application/pdf")

    # First upload succeeds
    EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[file1],
        uploader_id=u_id
    )

    # Second upload with identical bytes to same milestone must fail
    with pytest.raises(ConflictException, match="Duplicate evidence detected"):
        EvidenceService.upload_evidence_files(
            db=db_session,
            milestone_id=m_id,
            files=[(b"%PDF-1.4 Duplicate candidate document", "doc_renamed.pdf", "application/pdf")],
            uploader_id=u_id
        )


def test_evidence_upload_permission_enforcement(db_session, evidence_test_setup):
    """Verify non-org user is blocked from uploading evidence."""
    setup = evidence_test_setup
    m_id = setup["milestone"].id
    unauthorized_id = setup["other_user"].id

    file1 = (b"%PDF-1.4 Unauthorized document", "doc.pdf", "application/pdf")

    with pytest.raises(PermissionDeniedError, match="authorized NGO membership"):
        EvidenceService.upload_evidence_files(
            db=db_session,
            milestone_id=m_id,
            files=[file1],
            uploader_id=unauthorized_id
        )


def test_evidence_upload_batch_limit_enforcement(db_session, evidence_test_setup):
    """Verify uploading more files than batch limit is rejected."""
    setup = evidence_test_setup
    m_id = setup["milestone"].id
    u_id = setup["ngo_user"].id

    # 11 files with limit 10
    files = [(b"%PDF-1.4 File " + str(i).encode(), f"file_{i}.pdf", "application/pdf") for i in range(11)]

    with pytest.raises(ValidationException, match="exceeds maximum allowed"):
        EvidenceService.upload_evidence_files(
            db=db_session,
            milestone_id=m_id,
            files=files,
            uploader_id=u_id
        )
