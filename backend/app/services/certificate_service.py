"""
certificate_service.py - End-to-End Proof-of-Impact Certificate Generation & Lifecycle
======================================================================================
Coordinates multi-gate eligibility validation, RFC 8785 canonical claim construction,
Ed25519 digital signing, PDF/QR code generation, and append-only Impact Ledger recording.
Supports immutable status transitions: ISSUED, ACTIVE, SUSPENDED, REVOKED, SUPERSEDED.
"""

import uuid
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash
from backend.app.core.constants import (
    CertificateType, CertificateStatus, CertificateRevocationReason,
    ImpactLedgerEventType, DomainEventType
)
from backend.app.core.events import DomainEvent
from backend.app.services.outbox_service import OutboxService
from backend.app.db.models.financial import Donation
from backend.app.db.models.projects import Milestone, Project
from backend.app.db.models.users import Organization, User
from backend.app.db.models.reviews_impact import (
    ImpactCertificate, CertificateStatusEvent
)
from backend.app.services.certificate_eligibility_service import CertificateEligibilityService
from backend.app.services.key_management_service import KeyManagementService
from backend.app.services.certificate_rendering_service import CertificateRenderingService
from backend.app.services.impact_ledger_service import ImpactLedgerService


class CertificateService:
    @classmethod
    def issue_contribution_certificate(
        cls,
        db: Session,
        donation_id: str,
        actor_id: Optional[str] = None,
        public_attribution: bool = False,
        public_display_name: Optional[str] = None,
        recipient_name: Optional[str] = None,
        include_disbursement_claim: bool = False
    ) -> ImpactCertificate:
        """
        Validates eligibility, builds canonical claims, digitally signs with Ed25519,
        renders PDF/QR artifacts, appends to the Impact Ledger, and persists the certificate.
        """
        # 1. Run Multi-Gate Eligibility Engine
        cert_type = (
            CertificateType.DISBURSEMENT_ATTESTATION.value
            if include_disbursement_claim
            else CertificateType.CONTRIBUTION_IMPACT.value
        )
        eligibility = CertificateEligibilityService.evaluate_eligibility(
            db=db,
            donation_id=donation_id,
            certificate_type=cert_type,
            include_disbursement_claim=include_disbursement_claim
        )
        if not eligibility.is_eligible:
            raise ValueError(f"Certificate issuance ineligible: {eligibility.explanation}")

        # 2. Retrieve Domain Entities
        donation = db.query(Donation).filter(Donation.id == donation_id).first()
        milestone = db.query(Milestone).filter(Milestone.id == eligibility.milestone_id).first()
        project = db.query(Project).filter(Project.id == eligibility.project_id).first()
        org = db.query(Organization).filter(Organization.id == eligibility.organization_id).first()
        donor = db.query(User).filter(User.id == donation.donor_id).first() if donation.donor_id else None

        # 3. Generate Identifiers & Verification Token
        verification_id = secrets.token_urlsafe(32)
        date_str = datetime.utcnow().strftime("%Y%m%d")
        rand_suffix = uuid.uuid4().hex[:8].upper()
        certificate_number = f"CC-CERT-{date_str}-{rand_suffix}"
        idempotency_key = f"IDEMP-DONATION-{donation_id}-{cert_type}"

        # 4. Resolve Donor Privacy Display
        resolved_recipient = (
            recipient_name or
            getattr(donation, "donor_name", None) or
            (getattr(donor, "full_name", None) if donor else None) or
            "Anonymous Contributor"
        )
        if public_attribution:
            display_name = public_display_name or resolved_recipient
            is_anon = False
        else:
            display_name = "Anonymous Contributor"
            is_anon = True

        # 5. Build Canonical Claims Payload
        issuer_url = getattr(settings, "ISSUER_PUBLIC_URL", "http://localhost:8000")
        verify_base = getattr(settings, "FRONTEND_VERIFY_URL", "http://localhost:8501/?page=verify_certificate")
        verification_url = f"{issuer_url}/api/v1/public/certificates/{verification_id}"

        claims: Dict[str, Any] = {
            "schema_version": getattr(settings, "CERTIFICATE_SCHEMA_VERSION", "v9.0.0"),
            "certificate_type": cert_type,
            "certificate_number": certificate_number,
            "verification_id": verification_id,
            "issuer": {
                "id": getattr(settings, "ISSUER_ID", "charitychain-core-v1"),
                "name": getattr(settings, "ISSUER_NAME", "CharityChain Platform"),
                "public_url": issuer_url
            },
            "issued_at": datetime.utcnow().isoformat(),
            "donor": {
                "display_name": display_name,
                "is_anonymous": is_anon
            },
            "contribution": {
                "donation_id": str(donation.id),
                "amount_cents": eligibility.qualifying_amount_cents,
                "currency": eligibility.currency
            },
            "project": {
                "id": str(project.id) if project else "",
                "title": project.title if project else "Project"
            },
            "milestone": {
                "id": str(milestone.id) if milestone else "",
                "sequence_number": milestone.sequence_number if milestone else 1,
                "title": milestone.title if milestone else "Milestone"
            },
            "organization": {
                "id": str(org.id) if org else "",
                "name": org.name if org else "Verified Organization"
            },
            "verification_summary": {
                "review_case_id": eligibility.review_case_id,
                "snapshot_digest": eligibility.snapshot_digest or "GENESIS_SNAPSHOT",
                "dual_key_approved": True,
                "policy_version": eligibility.policy_version
            },
            "disbursement_attestation": {
                "disbursement_id": str(eligibility.disbursement_id),
                "status": "COMPLETED"
            } if eligibility.disbursement_id else None
        }

        # 6. Canonicalize & Digitally Sign Payload with Ed25519
        KeyManagementService.ensure_active_key_registered(db)
        canonical_str = canonicalize_json(claims)
        signature_b64, signing_key_id, payload_hash = KeyManagementService.sign_payload(canonical_str)

        # 7. Render PDF and QR Code Artifacts
        pdf_bytes, pdf_hash = CertificateRenderingService.render_certificate_pdf(
            claims=claims,
            signature_b64=signature_b64,
            signing_key_id=signing_key_id,
            verification_url=verification_url
        )
        pdf_storage_key = CertificateRenderingService.save_pdf_artifact(
            certificate_number=certificate_number,
            pdf_bytes=pdf_bytes
        )

        # 8. Persist Modernized ImpactCertificate Record
        certificate = ImpactCertificate(
            donation_id=donation.id,
            milestone_id=milestone.id if milestone else donation.milestone_id,
            project_id=project.id if project else donation.project_id,
            organization_id=org.id if org else None,
            review_case_id=eligibility.review_case_id,
            disbursement_id=eligibility.disbursement_id,
            certificate_number=certificate_number,
            verification_id=verification_id,
            idempotency_key=idempotency_key,
            certificate_type=cert_type,
            schema_version=getattr(settings, "CERTIFICATE_SCHEMA_VERSION", "v9.0.0"),
            recipient_name=resolved_recipient,
            public_attribution=public_attribution,
            public_display_name=public_display_name,
            canonical_payload=canonical_str,
            payload_hash=payload_hash,
            signature=signature_b64,
            signature_algorithm="Ed25519",
            signing_key_id=signing_key_id,
            issuer_id=getattr(settings, "ISSUER_ID", "charitychain-core-v1"),
            status=CertificateStatus.ISSUED.value,
            pdf_storage_key=pdf_storage_key,
            pdf_hash=pdf_hash,
            qr_code_url=verification_url,
            certificate_hash=payload_hash,
            token_uri=f"certificate://proof-of-impact/{verification_id}",
            metadata_json=canonical_str
        )
        db.add(certificate)
        db.flush()

        # 9. Record Initial Status Event
        status_event = CertificateStatusEvent(
            certificate_id=certificate.id,
            previous_status=CertificateStatus.PENDING_ELIGIBILITY.value,
            new_status=CertificateStatus.ISSUED.value,
            reason_code="INITIAL_ISSUANCE",
            reason_description="Certificate generated following valid milestone review approval.",
            actor_id=actor_id
        )
        db.add(status_event)

        # 10. Append to Milestone Impact Ledger Stream
        stream_id = f"milestone-{milestone.id}" if milestone else f"project-{project.id}"
        ImpactLedgerService.append_event(
            db=db,
            stream_id=stream_id,
            event_type=ImpactLedgerEventType.CERTIFICATE_ISSUED.value,
            source_event_type="CERTIFICATE_ISSUED",
            source_event_id=certificate.id,
            payload={
                "certificate_id": certificate.id,
                "certificate_number": certificate.certificate_number,
                "verification_id": certificate.verification_id,
                "certificate_type": certificate.certificate_type,
                "donation_id": str(donation.id),
                "qualifying_amount_cents": eligibility.qualifying_amount_cents,
                "payload_hash": payload_hash,
                "signing_key_id": signing_key_id
            },
            actor_id=actor_id,
            aggregate_type="MILESTONE",
            aggregate_id=str(milestone.id) if milestone else ""
        )

        # 11. Enqueue Outbox Event for Notifications & Webhooks
        event = DomainEvent.create(
            event_type=DomainEventType.CERTIFICATE_ISSUED.value,
            entity_type="impact_certificate",
            entity_id=certificate.id,
            organization_id=org.id if org else None,
            actor_user_id=actor_id,
            payload={
                "certificate_id": certificate.id,
                "certificate_number": certificate.certificate_number,
                "verification_id": certificate.verification_id,
                "verification_url": verification_url,
                "recipient_name": resolved_recipient,
                "donor_id": donation.donor_id if donation else None,
                "donor_email": donation.donor_email if donation else None,
                "project_title": project.title if project else "Project",
                "impact_summary": claims.get("verification_summary", {}).get("impact_summary") or "Milestone completed with verified impact."
            }
        )
        OutboxService.record_event(db, event)

        db.commit()
        db.refresh(certificate)
        return certificate

    @classmethod
    def suspend_certificate(
        cls,
        db: Session,
        certificate_id: str,
        reason_code: str,
        reason_description: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> ImpactCertificate:
        """Temporarily suspends an active certificate pending dispute or investigation."""
        cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == certificate_id).first()
        if not cert:
            raise ValueError(f"Certificate '{certificate_id}' not found.")

        prev_status = cert.status
        cert.status = CertificateStatus.SUSPENDED.value
        cert.revocation_reason = reason_code

        status_event = CertificateStatusEvent(
            certificate_id=cert.id,
            previous_status=prev_status,
            new_status=CertificateStatus.SUSPENDED.value,
            reason_code=reason_code,
            reason_description=reason_description or "Certificate suspended pending review.",
            actor_id=actor_id
        )
        db.add(status_event)

        stream_id = f"milestone-{cert.milestone_id}" if cert.milestone_id else "global-stream"
        ImpactLedgerService.append_event(
            db=db,
            stream_id=stream_id,
            event_type=ImpactLedgerEventType.CERTIFICATE_SUSPENDED.value,
            source_event_type="CERTIFICATE_SUSPENDED",
            source_event_id=cert.id,
            payload={
                "certificate_id": cert.id,
                "certificate_number": cert.certificate_number,
                "reason_code": reason_code,
                "reason_description": reason_description
            },
            actor_id=actor_id
        )

        event = DomainEvent.create(
            event_type=DomainEventType.CERTIFICATE_SUSPENDED.value,
            entity_type="impact_certificate",
            entity_id=cert.id,
            organization_id=cert.organization_id,
            actor_user_id=actor_id,
            payload={
                "certificate_id": cert.id,
                "certificate_number": cert.certificate_number,
                "reason_code": reason_code,
                "reason_description": reason_description
            }
        )
        OutboxService.record_event(db, event)

        db.commit()
        db.refresh(cert)
        return cert

    @classmethod
    def revoke_certificate(
        cls,
        db: Session,
        certificate_id: str,
        reason_code: str,
        reason_description: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> ImpactCertificate:
        """Permanently revokes a certificate following a full refund, overturned review, or compromise."""
        cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == certificate_id).first()
        if not cert:
            raise ValueError(f"Certificate '{certificate_id}' not found.")

        prev_status = cert.status
        cert.status = CertificateStatus.REVOKED.value
        cert.revocation_reason = reason_code
        cert.revoked_at = datetime.utcnow()

        status_event = CertificateStatusEvent(
            certificate_id=cert.id,
            previous_status=prev_status,
            new_status=CertificateStatus.REVOKED.value,
            reason_code=reason_code,
            reason_description=reason_description or "Certificate revoked under governance policy.",
            actor_id=actor_id
        )
        db.add(status_event)

        stream_id = f"milestone-{cert.milestone_id}" if cert.milestone_id else "global-stream"
        ImpactLedgerService.append_event(
            db=db,
            stream_id=stream_id,
            event_type=ImpactLedgerEventType.CERTIFICATE_REVOKED.value,
            source_event_type="CERTIFICATE_REVOKED",
            source_event_id=cert.id,
            payload={
                "certificate_id": cert.id,
                "certificate_number": cert.certificate_number,
                "reason_code": reason_code,
                "reason_description": reason_description,
                "revoked_at": cert.revoked_at.isoformat()
            },
            actor_id=actor_id
        )

        event = DomainEvent.create(
            event_type=DomainEventType.CERTIFICATE_REVOKED.value,
            entity_type="impact_certificate",
            entity_id=cert.id,
            organization_id=cert.organization_id,
            actor_user_id=actor_id,
            payload={
                "certificate_id": cert.id,
                "certificate_number": cert.certificate_number,
                "reason_code": reason_code,
                "reason_description": reason_description,
                "revoked_at": cert.revoked_at.isoformat()
            }
        )
        OutboxService.record_event(db, event)

        db.commit()
        db.refresh(cert)
        return cert

    @classmethod
    def supersede_certificate(
        cls,
        db: Session,
        old_certificate_id: str,
        new_certificate_id: str,
        reason_code: str = CertificateRevocationReason.SUPERSEDED_BY_REVISION.value,
        actor_id: Optional[str] = None
    ) -> ImpactCertificate:
        """Marks an older certificate as superseded by a newer revision."""
        old_cert = db.query(ImpactCertificate).filter(ImpactCertificate.id == old_certificate_id).first()
        if not old_cert:
            raise ValueError(f"Certificate '{old_certificate_id}' not found.")

        prev_status = old_cert.status
        old_cert.status = CertificateStatus.SUPERSEDED.value
        old_cert.superseded_by_id = new_certificate_id

        status_event = CertificateStatusEvent(
            certificate_id=old_cert.id,
            previous_status=prev_status,
            new_status=CertificateStatus.SUPERSEDED.value,
            reason_code=reason_code,
            reason_description=f"Superseded by certificate {new_certificate_id}.",
            actor_id=actor_id
        )
        db.add(status_event)

        stream_id = f"milestone-{old_cert.milestone_id}" if old_cert.milestone_id else "global-stream"
        ImpactLedgerService.append_event(
            db=db,
            stream_id=stream_id,
            event_type=ImpactLedgerEventType.CERTIFICATE_SUPERSEDED.value,
            source_event_type="CERTIFICATE_SUPERSEDED",
            source_event_id=old_cert.id,
            payload={
                "old_certificate_id": old_cert.id,
                "superseded_by_id": new_certificate_id,
                "reason_code": reason_code
            },
            actor_id=actor_id
        )

        db.commit()
        db.refresh(old_cert)
        return old_cert
