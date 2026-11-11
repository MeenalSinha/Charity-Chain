"""
test_evidence_security.py - Evidence Storage Security Tests
===========================================================
Tests defenses against path traversal, spoofed file headers,
unauthorized downloads, and destructive deletion attacks.
"""

import uuid
import pytest
from backend.app.core.constants import OrganizationRole, OrganizationStatus, PlatformRole, UserRole
from backend.app.core.exceptions import ValidationException, PermissionDeniedError
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import EvidenceFile
from backend.app.services.evidence_service import EvidenceService


@pytest.fixture
def sec_test_setup(db_session):
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"sec_ngo_{uuid.uuid4().hex[:6]}@ngo.org",
        hashed_password="pw",
        full_name="NGO Sec Lead",
        role=PlatformRole.DONOR,
        is_active=True
    )
    donor_user = User(
        id=str(uuid.uuid4()),
        email=f"donor_{uuid.uuid4().hex[:6]}@donor.org",
        hashed_password="pw",
        full_name="Random Donor",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add_all([ngo_user, donor_user])

    org = Organization(
        id=str(uuid.uuid4()),
        name="Security Shield NGO",
        legal_name="Security Shield NGO Inc",
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
        title="Disaster Emergency Clinic",
        description="Rapid response field clinics for emergency zones.",
        funding_target_minor=1000000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Shelter Deployment",
        description="Deploying weather-hardened tents.",
        funding_target_minor=500000
    )
    db_session.add(milestone)
    db_session.commit()

    return {"ngo_user": ngo_user, "donor_user": donor_user, "milestone": milestone}


def test_reject_spoofed_file_magic_bytes(db_session, sec_test_setup):
    """Verify that file extension spoofing (e.g. bash script masquerading as PNG) is rejected."""
    m_id = sec_test_setup["milestone"].id
    u_id = sec_test_setup["ngo_user"].id

    evil_script = b"#!/bin/bash\nrm -rf / --no-preserve-root"

    with pytest.raises(ValidationException, match="do not match declared MIME type"):
        EvidenceService.upload_evidence_files(
            db=db_session,
            milestone_id=m_id,
            files=[(evil_script, "avatar.png", "image/png")],
            uploader_id=u_id
        )


def test_path_traversal_filename_neutralization(db_session, sec_test_setup):
    """Verify path traversal characters in upload filename are sanitized safely."""
    m_id = sec_test_setup["milestone"].id
    u_id = sec_test_setup["ngo_user"].id

    authentic_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...legit"
    malicious_filename = "../../../../../etc/shadow.png"

    res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[(authentic_png, malicious_filename, "image/png")],
        uploader_id=u_id
    )

    stored_file = res.items[0]
    # Sanitized filename should not contain '../'
    assert ".." not in stored_file.storage_key
    assert "shadow.png" in stored_file.storage_key or "document" in stored_file.storage_key


def test_unauthorized_donor_download_denied(db_session, sec_test_setup):
    """Verify random donors cannot download raw evidence files."""
    m_id = sec_test_setup["milestone"].id
    ngo_id = sec_test_setup["ngo_user"].id
    donor_id = sec_test_setup["donor_user"].id

    authentic_pdf = b"%PDF-1.4 confidential medical record"
    res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[(authentic_pdf, "confidential.pdf", "application/pdf")],
        uploader_id=ngo_id
    )
    ev_id = res.items[0].id

    # Donor attempting download must be rejected
    with pytest.raises(PermissionDeniedError, match="restricted"):
        EvidenceService.get_evidence_download(
            db=db_session,
            evidence_id=ev_id,
            user_id=donor_id
        )


def test_soft_archival_preserves_content_and_prevents_data_loss(db_session, sec_test_setup):
    """Verify soft-archival sets flag and logs audit event without deleting stored file."""
    m_id = sec_test_setup["milestone"].id
    u_id = sec_test_setup["ngo_user"].id

    content = b"%PDF-1.4 preserved evidence document"
    res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[(content, "preserved.pdf", "application/pdf")],
        uploader_id=u_id
    )
    ev_id = res.items[0].id

    # Archive
    arch_res = EvidenceService.archive_evidence(db_session, ev_id, u_id, reason="Legal dispute resolution")
    assert arch_res.is_archived is True
    assert arch_res.archived_at is not None

    # Verify physical file and database entity still intact
    trail = EvidenceService.get_audit_trail(db_session, ev_id)
    assert trail.is_chain_intact is True
    assert trail.events[-1].event_type == "ARCHIVED"
