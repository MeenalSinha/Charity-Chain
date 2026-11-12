"""
test_certificate_idor.py - Security Tests for Certificate IDOR & Authorization Controls
========================================================================================
Verifies that non-admin users cannot access, download, or inspect audit history
for certificates belonging to other donors (IDOR prevention).
"""

import uuid
import pytest
from fastapi import HTTPException

from backend.app.core.constants import PlatformRole, CertificateStatus
from backend.app.db.models.users import User, AccountStatus
from backend.app.db.models.financial import Donation
from backend.app.db.models.reviews_impact import ImpactCertificate
from backend.app.api.v1.certificates import (
    get_certificate_detail,
    download_certificate_pdf,
    get_certificate_history
)


@pytest.fixture
def idor_fixture(db_session):
    donor_a = User(
        id=str(uuid.uuid4()),
        email=f"donor_a_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Donor Alice",
        hashed_password="mock",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    donor_b = User(
        id=str(uuid.uuid4()),
        email=f"donor_b_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Donor Bob",
        hashed_password="mock",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    admin = User(
        id=str(uuid.uuid4()),
        email=f"admin_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Admin Charlie",
        hashed_password="mock",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add_all([donor_a, donor_b, admin])
    db_session.commit()

    # Donation and certificate owned by Donor A
    don_a = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor_a.id,
        project_id=str(uuid.uuid4()),
        milestone_id=str(uuid.uuid4()),
        amount_minor_units=15000,
        currency="USD"
    )
    db_session.add(don_a)
    db_session.commit()

    cert_a = ImpactCertificate(
        id=str(uuid.uuid4()),
        donation_id=don_a.id,
        milestone_id=don_a.milestone_id,
        certificate_number="CC-IDOR-001",
        verification_id=f"v_id_{uuid.uuid4().hex[:16]}",
        idempotency_key=f"idem_{uuid.uuid4()}",
        certificate_type="CONTRIBUTION_IMPACT",
        status=CertificateStatus.ISSUED.value,
        recipient_name="Donor Alice",
        canonical_payload="{}",
        payload_hash="0" * 64,
        signature="mock_sig",
        signing_key_id="key-1"
    )
    db_session.add(cert_a)
    db_session.commit()

    return {
        "donor_a": donor_a,
        "donor_b": donor_b,
        "admin": admin,
        "cert_a": cert_a
    }


def test_donor_b_cannot_access_donor_a_certificate(db_session, idor_fixture):
    """Verifies that Donor B receives 403 Forbidden when requesting Donor A's certificate."""
    with pytest.raises(HTTPException) as exc_info:
        get_certificate_detail(
            certificate_id=idor_fixture["cert_a"].id,
            current_user=idor_fixture["donor_b"],
            db=db_session
        )
    assert exc_info.value.status_code == 403
    assert "Forbidden" in exc_info.value.detail


def test_donor_b_cannot_download_donor_a_certificate(db_session, idor_fixture):
    """Verifies that Donor B receives 403 Forbidden when downloading Donor A's PDF certificate."""
    with pytest.raises(HTTPException) as exc_info:
        download_certificate_pdf(
            certificate_id=idor_fixture["cert_a"].id,
            current_user=idor_fixture["donor_b"],
            db=db_session
        )
    assert exc_info.value.status_code == 403
    assert "Forbidden" in exc_info.value.detail


def test_donor_b_cannot_view_donor_a_certificate_history(db_session, idor_fixture):
    """Verifies that Donor B receives 403 Forbidden when requesting Donor A's certificate history."""
    with pytest.raises(HTTPException) as exc_info:
        get_certificate_history(
            certificate_id=idor_fixture["cert_a"].id,
            current_user=idor_fixture["donor_b"],
            db=db_session
        )
    assert exc_info.value.status_code == 403
    assert "Forbidden" in exc_info.value.detail


def test_donor_a_and_admin_have_access(db_session, idor_fixture):
    """Verifies that owner Donor A and Admin both successfully access Donor A's certificate."""
    # Donor A accessing own certificate
    res_a = get_certificate_detail(
        certificate_id=idor_fixture["cert_a"].id,
        current_user=idor_fixture["donor_a"],
        db=db_session
    )
    assert res_a.id == idor_fixture["cert_a"].id

    # Admin accessing Donor A's certificate
    res_admin = get_certificate_detail(
        certificate_id=idor_fixture["cert_a"].id,
        current_user=idor_fixture["admin"],
        db=db_session
    )
    assert res_admin.id == idor_fixture["cert_a"].id

