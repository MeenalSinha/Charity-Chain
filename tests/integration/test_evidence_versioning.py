"""
test_evidence_versioning.py - Multi-Versioning Integration Tests
================================================================
Verifies immutable history preservation, sequential versioning,
and parent-child audit chain linkages.
"""

import uuid
import pytest
from sqlalchemy import select

from backend.app.core.constants import OrganizationRole, OrganizationStatus, PlatformRole
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import EvidenceFile, EvidenceAuditEvent
from backend.app.services.evidence_service import EvidenceService


@pytest.fixture
def version_test_setup(db_session):
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"lead_{uuid.uuid4().hex[:6]}@ngo.org",
        hashed_password="pw",
        full_name="Project Lead",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Education Relief",
        legal_name="Global Education Relief Inc",
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
        title="Classroom Roofing Project",
        description="Replacing storm-damaged school roofs with corrugated galvanized iron.",
        funding_target_minor=800000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Roof Timber Framework Installation",
        description="Inspect and mount hardwood trusses.",
        funding_target_minor=400000
    )
    db_session.add(milestone)
    db_session.commit()

    return {"ngo_user": ngo_user, "org": org, "project": project, "milestone": milestone}


def test_evidence_version_lifecycle(db_session, version_test_setup):
    """Verify uploading version 2 creates linked record and preserves version 1 intact."""
    setup = version_test_setup
    m_id = setup["milestone"].id
    u_id = setup["ngo_user"].id

    # 1. Upload Version 1
    v1_content = b"\xff\xd8\xff\xe0JFIF original low-res photo"
    res1 = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[(v1_content, "roof_trusses.jpg", "image/jpeg")],
        uploader_id=u_id
    )
    v1_id = res1.items[0].id

    # 2. Upload Version 2 replacement
    v2_content = b"\xff\xd8\xff\xe0JFIF revised high-resolution photograph"
    v2_resp = EvidenceService.upload_evidence_version(
        db=db_session,
        parent_evidence_id=v1_id,
        file_bytes=v2_content,
        filename="roof_trusses_hd.jpg",
        mime_type="image/jpeg",
        uploader_id=u_id,
        reason="Updated with high-resolution inspection photos"
    )

    assert v2_resp.version_number == 2
    assert v2_resp.parent_evidence_id == v1_id
    assert v2_resp.original_filename == "roof_trusses_hd.jpg"

    # 3. Verify Version 1 still exists in database and on disk
    v1_entity = db_session.execute(select(EvidenceFile).where(EvidenceFile.id == v1_id)).scalar_one()
    assert v1_entity.version_number == 1
    assert v1_entity.file_hash != v2_resp.file_hash

    # 4. Verify version history endpoint output
    history = EvidenceService.get_version_history(db_session, v2_resp.id)
    assert history.current_version == 2
    assert len(history.versions) == 2
    assert history.versions[0].version_number == 1
    assert history.versions[1].version_number == 2

    # 5. Check audit trails for both
    trail_v1 = EvidenceService.get_audit_trail(db_session, v1_id)
    assert trail_v1.is_chain_intact is True
    # v1 has UPLOADED and SUPERSEDED
    event_types_v1 = [e.event_type for e in trail_v1.events]
    assert "UPLOADED" in event_types_v1
    assert "SUPERSEDED" in event_types_v1

    trail_v2 = EvidenceService.get_audit_trail(db_session, v2_resp.id)
    assert trail_v2.is_chain_intact is True
    assert trail_v2.events[0].event_type == "VERSION_CREATED"
