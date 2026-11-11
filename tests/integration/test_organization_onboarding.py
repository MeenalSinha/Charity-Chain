"""
test_organization_onboarding.py - Integration tests for Organization Onboarding & Verification
=============================================================================================
Tests atomic organization creation, document upload with magic byte check,
submission invariants, review queue, and reviewer decisions.
"""

import pytest
import io
from backend.app.core.constants import (
    OrganizationType, OrganizationStatus, OrganizationDocumentType,
    PlatformRole, OrganizationRole
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, UpdateOrganizationRequest,
    OrganizationDecisionRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.auth_service import AuthService
from backend.app.api.schemas import RegisterRequest
from backend.app.integrations.storage.local import LocalStorageProvider
from backend.app.core.exceptions import ValidationException, ForbiddenException
from backend.app.db.models.users import User, AccountStatus


@pytest.fixture
def onboarding_users(db_session):
    """Fixture providing an NGO applicant user, a reviewer, and a platform admin."""
    # 1. NGO Applicant
    ngo_user = AuthService.register_user(db_session, RegisterRequest(
        full_name="Alice Founder",
        email="alice.founder@ngo.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u1 = db_session.get(User, ngo_user.user_id)
    u1.is_verified = True
    u1.account_status = AccountStatus.ACTIVE
    
    # 2. Independent Reviewer
    rev_user = AuthService.register_user(db_session, RegisterRequest(
        full_name="Bob Reviewer",
        email="bob.reviewer@charitychain.org",
        password="SecurePassword123!",
        password_confirm="SecurePassword123!"
    ))
    u2 = db_session.get(User, rev_user.user_id)
    u2.is_verified = True
    u2.account_status = AccountStatus.ACTIVE
    u2.platform_role = PlatformRole.REVIEWER

    db_session.commit()
    return {
        "ngo_user_id": ngo_user.user_id,
        "reviewer_id": rev_user.user_id
    }


def test_atomic_organization_creation(db_session, onboarding_users):
    """Verify organization registration creates org in DRAFT and assigns creator as OWNER."""
    user_id = onboarding_users["ngo_user_id"]
    req = CreateOrganizationRequest(
        legal_name="Rainforest Protection Alliance",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="Brazilian Charity Registry",
        registration_number="BR-889922",
        country_of_registration="BR",
        registered_address="Avenida Paulista 1000, Sao Paulo",
        operating_address="Avenida Paulista 1000, Sao Paulo",
        primary_operating_region="Amazonas Basin",
        mission_statement="Preserving indigenous rainforest lands."
    )
    
    org = NGOOnboardingService.create_organization(db_session, req, user_id)
    assert org.id is not None
    assert org.legal_name == "Rainforest Protection Alliance"
    assert org.status == OrganizationStatus.DRAFT.value
    
    # Verify creator is OWNER
    user_orgs = NGOOnboardingService.list_user_organizations(db_session, user_id)
    assert len(user_orgs) == 1
    assert user_orgs[0].id == org.id


def test_document_upload_and_verification_submission(db_session, onboarding_users, tmp_path):
    """Verify document upload validates file bytes, calculates hash, and enables submission."""
    user_id = onboarding_users["ngo_user_id"]
    storage = LocalStorageProvider(str(tmp_path / "documents"))

    # Create org
    req = CreateOrganizationRequest(
        legal_name="Clean Oceans Initiative",
        organization_type=OrganizationType.CHARITY,
        registration_authority="UK Charity Commission",
        registration_number="UK-112233",
        country_of_registration="GB",
        registered_address="221B Baker St, London",
        operating_address="221B Baker St, London",
        primary_operating_region="North Sea",
        mission_statement="Removing plastics from marine ecosystems."
    )
    org = NGOOnboardingService.create_organization(db_session, req, user_id)

    # Submitting without documents must fail
    with pytest.raises(ValidationException) as exc_info:
        NGOOnboardingService.submit_for_verification(db_session, org.id, user_id)
    assert "verification document" in str(exc_info.value.message).lower()

    # Valid PDF magic bytes: %PDF-1.5
    valid_pdf_content = b"%PDF-1.5\n%Fake PDF content for automated test verification\n%%EOF"
    doc = NGOOnboardingService.upload_verification_document(
        db=db_session,
        organization_id=org.id,
        user_id=user_id,
        document_type=OrganizationDocumentType.REGISTRATION_CERTIFICATE.value,
        filename="incorporation_cert.pdf",
        content=valid_pdf_content,
        storage_provider=storage
    )
    assert doc.id is not None
    assert doc.original_filename == "incorporation_cert.pdf"
    assert len(doc.file_hash_sha256) == 64

    # Invalid executable content must be rejected
    invalid_exe_content = b"MZ\x90\x00\x03\x00\x00\x00This is an executable payload"
    with pytest.raises(ValidationException):
        NGOOnboardingService.upload_verification_document(
            db=db_session,
            organization_id=org.id,
            user_id=user_id,
            document_type=OrganizationDocumentType.TAX_EXEMPTION_CERTIFICATE.value,
            filename="malicious.exe",
            content=invalid_exe_content,
            storage_provider=storage
        )

    # Now submit for verification
    submitted_org = NGOOnboardingService.submit_for_verification(db_session, org.id, user_id)
    assert submitted_org.status == OrganizationStatus.PENDING.value


def test_reviewer_queue_and_approval_flow(db_session, onboarding_users, tmp_path):
    """Verify reviewer inspects pending organization and issues approval."""
    user_id = onboarding_users["ngo_user_id"]
    reviewer_id = onboarding_users["reviewer_id"]
    storage = LocalStorageProvider(str(tmp_path / "documents"))

    # Create & submit org
    req = CreateOrganizationRequest(
        legal_name="Solar For Schools",
        organization_type=OrganizationType.NGO,
        registration_authority="Kenyan NGO Board",
        registration_number="KE-554433",
        country_of_registration="KE",
        registered_address="Nairobi Tech Hub",
        operating_address="Nairobi Tech Hub",
        primary_operating_region="Rift Valley",
        mission_statement="Installing solar panels on rural public schools."
    )
    org = NGOOnboardingService.create_organization(db_session, req, user_id)
    
    valid_pdf_content = b"%PDF-1.4\n%Solar certificate content\n%%EOF"
    NGOOnboardingService.upload_verification_document(
        db=db_session,
        organization_id=org.id,
        user_id=user_id,
        document_type=OrganizationDocumentType.REGISTRATION_CERTIFICATE.value,
        filename="accreditation.pdf",
        content=valid_pdf_content,
        storage_provider=storage
    )
    NGOOnboardingService.submit_for_verification(db_session, org.id, user_id)

    # Reviewer queries review queue
    queue = NGOOnboardingService.get_review_queue(db_session, page=1, page_size=10)
    assert any(item.id == org.id for item in queue.items)

    # Reviewer approves
    decision = OrganizationDecisionRequest(
        decision="APPROVE",
        review_notes="All legal registration verified with Kenyan authorities."
    )
    reviewed_org = NGOOnboardingService.review_organization(db_session, org.id, decision, reviewer_id)
    assert reviewed_org.status == OrganizationStatus.VERIFIED.value
    assert reviewed_org.verified_at is not None
    assert reviewed_org.reviewer_id == reviewer_id
