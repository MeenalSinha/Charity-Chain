"""
test_ai_verification_security.py - Security & Abuse Defense Tests for AI Engine
================================================================================
Tests authorization boundaries, cross-organization isolation, prompt injection defenses,
and resource exhaustion limits (decompression bombs).
"""

import uuid
import pytest
from backend.app.core.constants import PlatformRole, OrganizationRole, OrganizationStatus
from backend.app.core.exceptions import PermissionDeniedError
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.verification_service import VerificationService
from backend.app.integrations.ai.yolo_provider import YoloObjectDetector


@pytest.fixture
def sec_ai_setup(db_session, sample_jpeg_bytes):
    ngo_user_a = User(
        id=str(uuid.uuid4()),
        email=f"ngo_a_{uuid.uuid4().hex[:6]}@orga.org",
        hashed_password="pw",
        full_name="NGO Lead A",
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
    db_session.add_all([ngo_user_a, donor_user])

    org_a = Organization(
        id=str(uuid.uuid4()),
        name="Org A Relief",
        legal_name="Org A Relief Ltd",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add(org_a)

    mem_a = OrganizationMembership(
        id=str(uuid.uuid4()),
        organization_id=org_a.id,
        user_id=ngo_user_a.id,
        role=OrganizationRole.ADMIN,
        is_active=True
    )
    db_session.add(mem_a)

    project_a = Project(
        id=str(uuid.uuid4()),
        organization_id=org_a.id,
        title="Protected Clinic Initiative",
        description="Confidential medical site.",
        funding_target_minor=1000000,
        currency="USD"
    )
    db_session.add(project_a)

    milestone_a = Milestone(
        id=str(uuid.uuid4()),
        project_id=project_a.id,
        sequence_number=1,
        title="Facility Construction",
        description="Foundation.",
        funding_target_minor=500000
    )
    db_session.add(milestone_a)
    db_session.commit()

    upload_res = EvidenceService.upload_evidence_files(
        db=db_session,
        milestone_id=milestone_a.id,
        files=[(sample_jpeg_bytes, "clinic.jpg", "image/jpeg")],
        uploader_id=ngo_user_a.id
    )
    ev_id = upload_res.items[0].id

    return {
        "ngo_user_a": ngo_user_a,
        "donor_user": donor_user,
        "evidence_id": ev_id
    }


def test_unauthorized_donor_cannot_trigger_verification(db_session, sec_ai_setup):
    """Verify unauthorized external donors cannot trigger AI verification jobs."""
    ev_id = sec_ai_setup["evidence_id"]
    donor = sec_ai_setup["donor_user"]

    with pytest.raises(PermissionDeniedError, match="restricted"):
        VerificationService.submit_verification_job(
            db=db_session,
            evidence_id=ev_id,
            user=donor
        )


def test_cross_organization_verification_denied(db_session, sec_ai_setup, sample_jpeg_bytes):
    """Verify users from Org B cannot run verification on Org A's evidence."""
    ev_id = sec_ai_setup["evidence_id"]

    # Create Org B user
    ngo_user_b = User(
        id=str(uuid.uuid4()),
        email=f"ngo_b_{uuid.uuid4().hex[:6]}@orgb.org",
        hashed_password="pw",
        full_name="NGO Lead B",
        role=PlatformRole.DONOR,
        is_active=True
    )
    org_b = Organization(id=str(uuid.uuid4()), name="Org B", legal_name="Org B Inc", status=OrganizationStatus.VERIFIED)
    db_session.add_all([ngo_user_b, org_b])
    mem_b = OrganizationMembership(organization_id=org_b.id, user_id=ngo_user_b.id, role=OrganizationRole.ADMIN, is_active=True)
    db_session.add(mem_b)
    db_session.commit()

    with pytest.raises(PermissionDeniedError, match="restricted"):
        VerificationService.submit_verification_job(
            db=db_session,
            evidence_id=ev_id,
            user=ngo_user_b
        )


def test_decompression_bomb_pixels_limit_rejected():
    """Verify images exceeding MAX_IMAGE_PIXELS are blocked from YOLO inference."""
    from PIL import Image
    from io import BytesIO
    detector = YoloObjectDetector()

    # Create synthetic metadata claiming 10,000 x 5,000 pixels (50,000,000 pixels > 25,000,000 limit)
    img = Image.new("RGB", (6000, 5000), color=(100, 100, 100))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    large_bytes = buffer.getvalue()

    res = detector.detect_objects(large_bytes)
    assert res.passed is False
    assert res.status.value == "ERROR"
    assert "safe limit" in res.message.lower()

