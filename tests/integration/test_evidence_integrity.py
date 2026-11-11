"""
test_evidence_integrity.py - Cryptographic Content Verification Integration Tests
=================================================================================
Verifies live recalculation of SHA-256 hashes against stored assets,
tamper detection on corrupted files, and audit trail updates.
"""

import uuid
import pytest
from pathlib import Path
from sqlalchemy import select

from backend.app.core.constants import OrganizationRole, OrganizationStatus, PlatformRole
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.evidence import EvidenceFile, EvidenceAuditEvent
from backend.app.services.evidence_service import EvidenceService
from backend.app.integrations.storage.factory import get_storage_provider


@pytest.fixture
def integrity_test_setup(db_session):
    ngo_user = User(
        id=str(uuid.uuid4()),
        email=f"integ_{uuid.uuid4().hex[:6]}@ngo.org",
        hashed_password="pw",
        full_name="Integrity Auditor",
        role=PlatformRole.DONOR,
        is_active=True
    )
    db_session.add(ngo_user)

    org = Organization(
        id=str(uuid.uuid4()),
        name="Medical Relief Alliance",
        legal_name="Medical Relief Alliance Inc",
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
        title="Mobile Clinic Deployment",
        description="Equipping rural outreach vans with diagnostic ultrasound kits.",
        funding_target_minor=500000,
        currency="USD"
    )
    db_session.add(project)

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Diagnostic Equipment Procurement",
        description="Procure certified portable ultrasound scanners.",
        funding_target_minor=250000
    )
    db_session.add(milestone)
    db_session.commit()

    return {"ngo_user": ngo_user, "milestone": milestone}


def test_evidence_integrity_verification_positive_and_tampered(db_session, integrity_test_setup):
    """Verify cryptographic check passes on authentic file and fails when content is corrupted."""
    setup = integrity_test_setup
    m_id = setup["milestone"].id
    u_id = setup["ngo_user"].id

    authentic_content = b"%PDF-1.7 Valid ultrasound equipment receipt"
    upload_res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=m_id,
        files=[(authentic_content, "receipt.pdf", "application/pdf")],
        uploader_id=u_id
    )
    ev_id = upload_res.items[0].id

    # 1. Positive check on authentic file
    check1 = EvidenceService.verify_evidence_integrity(db_session, ev_id, actor_id=u_id)
    assert check1.is_valid is True
    assert check1.expected_hash == check1.computed_hash

    # Check that audit log recorded INTEGRITY_VERIFIED
    audits = db_session.execute(
        select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == ev_id).order_by(EvidenceAuditEvent.created_at.asc())
    ).scalars().all()
    assert len(audits) == 2
    assert audits[1].event_type == "INTEGRITY_VERIFIED"

    # 2. Corrupt the file on disk to simulate bit rot or tampering
    storage = get_storage_provider()
    ev_entity = db_session.execute(select(EvidenceFile).where(EvidenceFile.id == ev_id)).scalar_one()
    # Resolve file on local disk and write corrupted content
    if hasattr(storage, "_resolve_safe_path"):
        target_path = storage._resolve_safe_path(ev_entity.storage_key)
        target_path.write_bytes(b"%PDF-1.7 CORRUPTED CONTENT REPLACED ON DISK")

        # 3. Re-run integrity check
        check2 = EvidenceService.verify_evidence_integrity(db_session, ev_id, actor_id=u_id)
        assert check2.is_valid is False
        assert check2.expected_hash != check2.computed_hash

        # Check audit log recorded INTEGRITY_TAMPERED
        audits_after = db_session.execute(
            select(EvidenceAuditEvent).where(EvidenceAuditEvent.evidence_id == ev_id).order_by(EvidenceAuditEvent.created_at.asc())
        ).scalars().all()
        assert len(audits_after) == 3
        assert audits_after[2].event_type == "INTEGRITY_TAMPERED"

    # 4. Verify audit hash chain remains mathematically continuous
    trail = EvidenceService.get_audit_trail(db_session, ev_id)
    assert trail.is_chain_intact is True
