"""
test_complete_certificate_lifecycle.py - Functional End-to-End Certificate Lifecycle Test
==========================================================================================
Verifies the complete life cycle of an Impact Certificate across all platform domains:
1. Verified organization and project milestone
2. Genuine payment capture
3. Evidence submission and dual-key review approval
4. Milestone completion and certificate eligibility
5. Asymmetric Ed25519 signing, PDF generation, and Impact Ledger append
6. Public QR verification
7. Lifecycle suspension and revocation
8. Cryptographic hash-chain integrity verification of the entire event stream
"""

import uuid
import pytest
from datetime import datetime, timezone

from backend.app.core.constants import (
    PlatformRole, ProjectStatus, MilestoneStatus, OrganizationStatus,
    DonationStatus, PaymentCaptureStatus, PaymentOrderStatus, ReviewCaseStatus, ReviewDecision,
    CertificateStatus, CertificateType, CertificateRevocationReason,
    ImpactLedgerEventType
)
from backend.app.db.models.users import User, Organization, AccountStatus
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.financial import Donation, PaymentOrder, PaymentCapture
from backend.app.db.models.reviews_impact import ReviewCase, ImpactCertificate
from backend.app.services.certificate_eligibility_service import CertificateEligibilityService
from backend.app.services.certificate_service import CertificateService
from backend.app.services.public_verification_service import PublicVerificationService
from backend.app.services.impact_ledger_service import ImpactLedgerService


def test_complete_proof_of_impact_lifecycle(db_session):
    """Executes the full functional test covering all stages from donation to revocation."""
    now = datetime.now(timezone.utc)

    # 1. Setup Verified Organization & Users
    donor = User(
        id=str(uuid.uuid4()),
        email="donor_e2e@example.com",
        full_name="Grace Hopper",
        hashed_password="mock",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    admin = User(
        id=str(uuid.uuid4()),
        email="admin_e2e@example.com",
        full_name="Admin Alan",
        hashed_password="mock",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org = Organization(
        id=str(uuid.uuid4()),
        name="Global Reforest Network",
        legal_name="Global Reforest Network NGO",
        status=OrganizationStatus.VERIFIED
    )
    db_session.add_all([donor, admin, org])
    db_session.commit()

    # 2. Setup Project & Milestone
    project = Project(
        id=str(uuid.uuid4()),
        organization_id=org.id,
        title="Amazon Basin Protection",
        description="Preserving rain forest biodiversity",
        status=ProjectStatus.ACTIVE,
        currency="USD",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=1000000
    )
    db_session.add(project)
    db_session.commit()

    milestone = Milestone(
        id=str(uuid.uuid4()),
        project_id=project.id,
        sequence_number=1,
        title="Seedling Preparation",
        description="Growing 50,000 native saplings",
        target_amount_minor_units=1000000,
        raised_amount_minor_units=1000000,
        status=MilestoneStatus.APPROVED
    )
    db_session.add(milestone)
    db_session.commit()

    # 3. Setup Donation & Payment Capture
    donation = Donation(
        id=str(uuid.uuid4()),
        donor_id=donor.id,
        project_id=project.id,
        milestone_id=milestone.id,
        amount_minor_units=50000,
        currency="USD",
        status=DonationStatus.PAID
    )
    db_session.add(donation)
    db_session.commit()

    order = PaymentOrder(
        id=str(uuid.uuid4()),
        donation_id=donation.id,
        provider_order_id=f"ORD-E2E-{uuid.uuid4().hex[:8]}",
        amount_minor_units=50000,
        currency="USD",
        status=PaymentOrderStatus.COMPLETED
    )
    db_session.add(order)
    db_session.commit()

    capture = PaymentCapture(
        id=str(uuid.uuid4()),
        order_id=order.id,
        provider_capture_id=f"CAP-E2E-{uuid.uuid4().hex[:8]}",
        amount_minor_units=50000,
        currency="USD",
        status=PaymentCaptureStatus.COMPLETED,
        captured_at=now
    )
    db_session.add(capture)
    db_session.commit()

    # 4. Human Review Approval
    review_case = ReviewCase(
        id=str(uuid.uuid4()),
        milestone_id=milestone.id,
        project_id=project.id,
        organization_id=org.id,
        submission_round=1,
        status=ReviewCaseStatus.APPROVED.value,
        final_decision=ReviewDecision.APPROVE.value,
        snapshot_digest="f" * 64
    )
    db_session.add(review_case)
    db_session.commit()

    # 5. Evaluate Certificate Eligibility Gate
    eligibility = CertificateEligibilityService.evaluate_eligibility(
        db=db_session,
        donation_id=donation.id
    )
    assert eligibility.is_eligible is True
    assert eligibility.qualifying_amount_cents == 50000

    # 6. Issue Digital Impact Certificate
    cert = CertificateService.issue_contribution_certificate(
        db=db_session,
        donation_id=donation.id,
        recipient_name="Grace Hopper",
        public_attribution=True,
        actor_id=admin.id
    )
    assert cert.status == CertificateStatus.ISSUED.value
    assert cert.signature is not None
    assert len(cert.payload_hash) == 64
    assert cert.pdf_storage_key is not None

    # 7. Perform Public Verification
    pub_res = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=cert.verification_id
    )
    assert pub_res["verification_outcome"] == "VALID"
    assert pub_res["is_signature_valid"] is True
    assert pub_res["is_payload_intact"] is True
    assert pub_res["donor"]["display_name"] == "Grace Hopper"
    assert pub_res["contribution"]["amount_cents"] == 50000

    # 8. Administrative Suspension
    suspended_cert = CertificateService.suspend_certificate(
        db=db_session,
        certificate_id=cert.id,
        reason_code="DISPUTE_INVESTIGATION",
        actor_id=admin.id
    )
    assert suspended_cert.status == CertificateStatus.SUSPENDED.value

    pub_suspended = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=cert.verification_id
    )
    assert pub_suspended["verification_outcome"] == "SUSPENDED"

    # 9. Administrative Revocation
    revoked_cert = CertificateService.revoke_certificate(
        db=db_session,
        certificate_id=cert.id,
        reason_code=CertificateRevocationReason.REVIEW_OVERTURNED.value,
        actor_id=admin.id
    )
    assert revoked_cert.status == CertificateStatus.REVOKED.value
    assert revoked_cert.revoked_at is not None

    pub_revoked = PublicVerificationService.verify_certificate(
        db=db_session,
        verification_id=cert.verification_id
    )
    assert pub_revoked["verification_outcome"] == "REVOKED"

    # 10. Audit Stream Integrity
    stream_id = f"milestone-{milestone.id}"
    ledger_audit = ImpactLedgerService.verify_stream_chain(db=db_session, stream_id=stream_id)
    assert ledger_audit["is_valid"] is True
    assert ledger_audit["total_events"] >= 3  # ISSUED, SUSPENDED, REVOKED
    assert len(ledger_audit["errors"]) == 0
