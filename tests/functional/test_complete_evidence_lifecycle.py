"""
test_complete_evidence_lifecycle.py - End-to-End Functional Test for Evidence Lifecycle
========================================================================================
Validates the complete evidence lifecycle from multi-file upload,
cryptographic verification, versioning, authorized download, soft-archival,
and mathematical proof of unbroken audit chain.
"""

import uuid
import pytest
from sqlalchemy import select

from backend.app.core.constants import OrganizationRole, OrganizationStatus, PlatformRole, UserRole
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import EvidenceFile, EvidenceAuditEvent
from backend.app.services.evidence_service import EvidenceService


def test_complete_end_to_end_evidence_lifecycle(db_session):
    """
    Simulates a full production evidence lifecycle:
    1. Organization, project, and milestone setup
    2. Multi-file upload batch by NGO lead
    3. Live cryptographic integrity verification
    4. Authorized reviewer download
    5. Version replacement submission
    6. Soft-archival of obsolete record
    7. Continuous cryptographic audit chain verification
    """
    # 1. Setup actors & entities
    ngo_lead = User(
        id=str(uuid.uuid4()),
        email=f"director_{uuid.uuid4().hex[:6]}@aid.org",
        hashed_password="pw",
        full_name="Program Director",
        role=PlatformRole.DONOR,
        is_active=True
    )
    reviewer = User(
        id=str(uuid.uuid4()),
        email=f"reviewer_{uuid.uuid4().hex[:6]}@charitychain.org",
        hashed_password="pw",
        full_name="Independent Forensic Reviewer",
        role=PlatformRole.REVIEWER,
        is_active=True
    )
    db_session.add_all([ngo_lead, reviewer])

    org = Organization(
        id=str(uuid.uuid4()),
        name="Amazonian Reforestation Trust",
        legal_name="Amazonian Reforestation Trust Ltd",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org)

    mem = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        user_id=ngo_lead.id,
        role=OrganizationRole.OWNER,
        is_active=True
    )
    db_session.add(mem)

    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Canopy Restoration Sector 4",
        description="Planting 25,000 indigenous hardwood saplings across degraded pastureland.",
        funding_target_minor=2000000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Nursery Seedling Propagation",
        description="Germinate and pot 25,000 native saplings in shadehouses.",
        funding_target_minor=1000000
    )
    db_session.add(milestone)
    db_session.commit()

    # 2. NGO Lead uploads 2 evidence files in batch
    file1_bytes = b"%PDF-1.5 Forestry nursery audit and inventory sign-off"
    file2_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR Sapling canopy photo"
    
    batch = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone.id,
        files=[
            (file1_bytes, "nursery_audit.pdf", "application/pdf"),
            (file2_bytes, "canopy_overview.png", "image/png")
        ],
        uploader_id=ngo_lead.id
    )

    assert batch.total_uploaded == 2
    ev1_id = batch.items[0].id
    ev2_id = batch.items[1].id

    # 3. Cryptographic integrity verification
    chk1 = EvidenceService.verify_evidence_integrity(db_session, ev1_id, actor_id=ngo_lead.id)
    assert chk1.is_valid is True

    chk2 = EvidenceService.verify_evidence_integrity(db_session, ev2_id, actor_id=ngo_lead.id)
    assert chk2.is_valid is True

    # 4. Authorized reviewer download
    dl_url, dl_bytes, fname, mtype = EvidenceService.get_evidence_download(
        db=db_session,
        evidence_id=ev1_id,
        user_id=reviewer.id
    )
    assert fname == "nursery_audit.pdf"
    assert mtype == "application/pdf"
    if dl_bytes:
        assert dl_bytes == file1_bytes

    # 5. Version replacement submission (uploading v2 of ev2)
    revised_photo = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR High-definition drone mosaic"
    v2_ev = EvidenceService.upload_evidence_version(
        db=db_session,
        parent_evidence_id=ev2_id,
        file_bytes=revised_photo,
        filename="drone_mosaic_v2.png",
        mime_type="image/png",
        uploader_id=ngo_lead.id,
        reason="Upgraded with 4K multispectral drone survey"
    )

    assert v2_ev.version_number == 2
    assert v2_ev.parent_evidence_id == ev2_id

    # 6. Soft-archival
    archived = EvidenceService.archive_evidence(db_session, ev2_id, ngo_lead.id, reason="Superseded by drone mosaic v2")
    assert archived.is_archived is True

    # 7. Unbroken mathematical audit chain validation
    trail1 = EvidenceService.get_audit_trail(db_session, ev1_id)
    assert trail1.is_chain_intact is True
    # ev1 has: UPLOADED, INTEGRITY_VERIFIED, DOWNLOADED
    assert trail1.total_events == 3

    trail2 = EvidenceService.get_audit_trail(db_session, ev2_id)
    assert trail2.is_chain_intact is True
    # ev2 has: UPLOADED, INTEGRITY_VERIFIED, SUPERSEDED, ARCHIVED
    assert trail2.total_events == 4

    trail_v2 = EvidenceService.get_audit_trail(db_session, v2_ev.id)
    assert trail_v2.is_chain_intact is True
    assert trail_v2.events[0].event_type == "VERSION_CREATED"
