"""
ngo_onboarding_service.py - NGO Organization Onboarding & Verification Domain Service
======================================================================================
Manages organization registration with atomic OWNER creation, organization profile
management, jurisdiction-aware document handling, verification review queues,
and conflict-of-interest protected reviewer workflows.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import json

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func

from backend.app.core.config import settings
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, OrganizationStatus, OrganizationType,
    OrganizationDocumentType, DocumentReviewStatus, DocumentAccessClassification
)
from backend.app.core.exceptions import (
    ConflictException, NotFoundException, ValidationException, ForbiddenException
)
from backend.app.core.logging import logger
from backend.app.core.state_machines import StateMachineValidator
from backend.app.db.models.users import (
    User, Organization, OrganizationMembership, OrganizationDocument, NGOProfile
)
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.integrations.storage.local import document_storage
from backend.app.api.schemas import (
    CreateOrganizationRequest, UpdateOrganizationRequest,
    OrganizationPublicResponse, OrganizationDetailResponse,
    OrganizationDecisionRequest, OrganizationDocumentResponse,
    PaginationMeta, PaginatedResponse
)


class NGOOnboardingService:
    """Domain service for NGO organization registration, verification, and document management."""

    @staticmethod
    def create_organization(
        db: Session,
        arg2: Any,
        arg3: Any = None
    ) -> OrganizationDetailResponse:
        """
        Creates a new organization and assigns the creator as initial OWNER
        within an atomic database transaction.
        """
        if isinstance(arg2, CreateOrganizationRequest):
            req = arg2
            creator_user_id = str(arg3)
        else:
            creator_user_id = str(arg2)
            req = arg3
        user = db.execute(select(User).where(User.id == creator_user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException("Creator user not found.")

        # Check for unique organization name
        existing_name = db.execute(
            select(Organization).where(func.lower(Organization.name) == req.name.strip().lower())
        ).scalar_one_or_none()
        if existing_name:
            raise ConflictException(f"Organization name '{req.name}' is already registered.")

        # Check for unique registration number if supplied
        if req.registration_number:
            existing_reg = db.execute(
                select(Organization).where(Organization.registration_number == req.registration_number.strip())
            ).scalar_one_or_none()
            if existing_reg:
                raise ConflictException(f"Registration number '{req.registration_number}' is already registered.")

        # Encode JSON fields
        focus_json = json.dumps(req.focus_areas) if req.focus_areas else None
        regions_json = json.dumps(req.operating_regions) if req.operating_regions else None
        beneficiary_json = json.dumps(req.beneficiary_categories) if req.beneficiary_categories else None

        org = Organization(
            name=req.name.strip(),
            legal_name=req.legal_name.strip() if req.legal_name else req.name.strip(),
            display_name=req.display_name.strip() if req.display_name else req.name.strip(),
            description=req.description,
            organization_type=req.organization_type,
            country=req.country.upper() if req.country else "USA",
            registration_number=req.registration_number.strip() if req.registration_number else None,
            registration_authority=req.registration_authority,
            registration_date=req.registration_date,
            tax_id=req.tax_id,
            classification=req.classification,
            registered_address=req.registered_address,
            operating_address=req.operating_address,
            official_email=req.official_email.lower().strip() if req.official_email else user.email,
            official_phone=req.official_phone,
            website=req.website,
            mission_statement=req.mission_statement,
            vision=req.vision,
            focus_areas=focus_json,
            operating_regions=regions_json,
            beneficiary_categories=beneficiary_json,
            years_of_operation=req.years_of_operation or 0,
            status=OrganizationStatus.DRAFT
        )
        db.add(org)
        db.flush()

        # Atomic OWNER membership creation
        owner_membership = OrganizationMembership(
            user_id=creator_user_id,
            organization_id=org.id,
            role=OrganizationRole.OWNER,
            is_active=True
        )
        db.add(owner_membership)

        # Initial NGOProfile record
        ngo_profile = NGOProfile(
            organization_id=org.id,
            mission_statement=req.mission_statement,
            reputation_score=100
        )
        db.add(ngo_profile)
        db.flush()

        # Audit Event
        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action="ORGANIZATION_CREATED",
            actor_id=creator_user_id,
            after_state={"name": org.name, "owner_id": creator_user_id, "status": org.status.value}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Organization '{org.name}' ({org.id}) created by user {creator_user_id} with initial OWNER role.")
        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def get_organization(
        db: Session,
        org_id: str,
        requesting_user_id: Optional[str] = None
    ) -> Any:
        """
        Retrieves organization information.
        Returns OrganizationDetailResponse for authorized members/reviewers;
        Returns OrganizationPublicResponse for anonymous or general donors.
        """
        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        # Determine authorization
        is_privileged = False
        if requesting_user_id:
            user = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
            if user and user.role in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
                is_privileged = True
            else:
                mem = db.execute(
                    select(OrganizationMembership).where(
                        and_(
                            OrganizationMembership.organization_id == org_id,
                            OrganizationMembership.user_id == requesting_user_id,
                            OrganizationMembership.is_active == True
                        )
                    )
                ).scalar_one_or_none()
                if mem and mem.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
                    is_privileged = True

        if is_privileged:
            return NGOOnboardingService._build_detail_response(org)
        elif requesting_user_id:
            raise ForbiddenException("Access denied: You are not authorized to view details of this organization.")
        else:
            return NGOOnboardingService._build_public_response(org)

    @staticmethod
    def update_organization(
        db: Session,
        org_id: str,
        requesting_user_id: str,
        req: UpdateOrganizationRequest
    ) -> OrganizationDetailResponse:
        """Updates organization details. Requires OWNER or ADMIN role."""
        org, mem = NGOOnboardingService._get_org_and_verify_membership(
            db, org_id, requesting_user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        if req.legal_name is not None:
            org.legal_name = req.legal_name.strip()
        if req.display_name is not None:
            org.display_name = req.display_name.strip()
        if req.description is not None:
            org.description = req.description
        if req.organization_type is not None:
            org.organization_type = req.organization_type
        if req.country is not None:
            org.country = req.country.upper()
        if req.registration_number is not None:
            org.registration_number = req.registration_number.strip()
        if req.registration_authority is not None:
            org.registration_authority = req.registration_authority
        if req.registration_date is not None:
            org.registration_date = req.registration_date
        if req.tax_id is not None:
            org.tax_id = req.tax_id
        if req.classification is not None:
            org.classification = req.classification
        if req.registered_address is not None:
            org.registered_address = req.registered_address
        if req.operating_address is not None:
            org.operating_address = req.operating_address
        if req.official_email is not None:
            org.official_email = req.official_email.lower().strip()
        if req.official_phone is not None:
            org.official_phone = req.official_phone
        if req.website is not None:
            org.website = req.website
        if req.mission_statement is not None:
            org.mission_statement = req.mission_statement
        if req.vision is not None:
            org.vision = req.vision
        if req.focus_areas is not None:
            org.focus_areas = json.dumps(req.focus_areas)
        if req.operating_regions is not None:
            org.operating_regions = json.dumps(req.operating_regions)
        if req.beneficiary_categories is not None:
            org.beneficiary_categories = json.dumps(req.beneficiary_categories)
        if req.years_of_operation is not None:
            org.years_of_operation = req.years_of_operation

        db.flush()
        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action="ORGANIZATION_UPDATED",
            actor_id=requesting_user_id,
            after_state={"name": org.name, "updated_by": requesting_user_id}
        )
        db.add(audit)
        db.commit()

        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def upload_document(
        db: Session,
        org_id: str,
        user_id: str,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        document_type: OrganizationDocumentType,
        expires_at: Optional[datetime] = None,
        storage_provider: Optional[LocalStorageProvider] = None
    ) -> OrganizationDocumentResponse:
        """Uploads and registers an onboarding verification document."""
        org, mem = NGOOnboardingService._get_org_and_verify_membership(
            db, org_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        storage = storage_provider or document_storage
        # Store file safely via local document storage
        storage_key, relative_url, sha256_hash, file_size = storage.save_file_sync(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            subfolder=org_id
        )

        doc = OrganizationDocument(
            organization_id=org_id,
            document_type=document_type,
            original_filename=filename,
            storage_filename=storage_key.split("/")[-1],
            storage_path=storage_key,
            mime_type=mime_type,
            file_size_bytes=file_size,
            file_hash_sha256=sha256_hash,
            uploaded_by_id=user_id,
            version_number=1,
            expires_at=expires_at,
            review_status=DocumentReviewStatus.PENDING,
            access_classification=DocumentAccessClassification.CONFIDENTIAL
        )
        db.add(doc)
        db.flush()

        audit = AuditEvent(
            entity_name="OrganizationDocument",
            entity_id=doc.id,
            action="ORG_DOCUMENT_UPLOADED",
            actor_id=user_id,
            after_state={"organization_id": org_id, "doc_type": document_type.value, "file_hash": sha256_hash}
        )
        db.add(audit)
        db.commit()

        return NGOOnboardingService._build_document_response(doc)

    @staticmethod
    def upload_verification_document(
        db: Session,
        organization_id: str,
        user_id: str,
        document_type: Any,
        filename: str,
        content: bytes,
        storage_provider: Optional[LocalStorageProvider] = None
    ) -> OrganizationDocumentResponse:
        """Convenience method for uploading verification documents."""
        lower = filename.lower()
        if lower.endswith(".pdf"):
            mime = "application/pdf"
        elif lower.endswith(".png"):
            mime = "image/png"
        elif lower.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        else:
            mime = "application/octet-stream"

        doc_type_val = document_type.value if hasattr(document_type, "value") else str(document_type)
        try:
            resolved_doc_type = OrganizationDocumentType(doc_type_val)
        except Exception:
            resolved_doc_type = OrganizationDocumentType.REGISTRATION_CERTIFICATE

        return NGOOnboardingService.upload_document(
            db=db,
            org_id=organization_id,
            user_id=user_id,
            file_bytes=content,
            filename=filename,
            mime_type=mime,
            document_type=resolved_doc_type,
            storage_provider=storage_provider
        )

    @staticmethod
    def list_documents(
        db: Session,
        org_id: str,
        requesting_user_id: str
    ) -> List[OrganizationDocumentResponse]:
        """Lists verification documents for an organization. Restricted to org members or reviewers."""
        user = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
        is_platform_reviewer = user and user.role in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)

        if not is_platform_reviewer:
            NGOOnboardingService._get_org_and_verify_membership(
                db, org_id, requesting_user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER)
            )

        docs = db.execute(
            select(OrganizationDocument).where(
                and_(
                    OrganizationDocument.organization_id == org_id,
                    OrganizationDocument.is_active == True
                )
            ).order_by(OrganizationDocument.created_at.desc())
        ).scalars().all()

        return [NGOOnboardingService._build_document_response(d) for d in docs]

    @staticmethod
    def submit_verification(
        db: Session,
        org_id: str,
        requesting_user_id: str
    ) -> OrganizationDetailResponse:
        """Submits organization application for reviewer examination."""
        org, mem = NGOOnboardingService._get_org_and_verify_membership(
            db, org_id, requesting_user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        # Validate that required onboarding data exists
        if not org.registration_number:
            raise ValidationException("Legal registration number is required prior to verification submission.")
        if not org.mission_statement:
            raise ValidationException("Mission statement is required prior to verification submission.")

        # Require at least one active document
        doc_count = db.execute(
            select(func.count(OrganizationDocument.id)).where(
                and_(
                    OrganizationDocument.organization_id == org_id,
                    OrganizationDocument.is_active == True
                )
            )
        ).scalar() or 0

        if doc_count == 0:
            raise ValidationException("At least one registration or nonprofit verification document is required.")

        # Validate lifecycle transition
        StateMachineValidator.validate_organization_transition(
            current=org.status,
            target=OrganizationStatus.PENDING,
            actor_org_role=mem.role
        )

        org.status = OrganizationStatus.PENDING
        org.submitted_at = datetime.utcnow()
        db.flush()

        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action="ORGANIZATION_SUBMITTED_FOR_REVIEW",
            actor_id=requesting_user_id,
            after_state={"status": org.status.value, "submitted_at": org.submitted_at.isoformat()}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Organization '{org.name}' ({org.id}) submitted for verification by {requesting_user_id}.")
        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def review_organization(
        db: Session,
        org_id: Optional[str] = None,
        arg3: Any = None,
        arg4: Any = None,
        organization_id: Optional[str] = None,
        decision: Any = None,
        reviewer_id: Optional[str] = None,
        req: Any = None
    ) -> OrganizationDetailResponse:
        """
        Executes a reviewer decision (APPROVE, REJECT, REQUEST_REVISION).
        Enforces Conflict-of-Interest prevention: Reviewers cannot review organizations they belong to.
        """
        target_org_id = org_id or organization_id
        target_req = req or decision
        target_reviewer_id = reviewer_id

        if target_req is None and target_reviewer_id is None:
            if isinstance(arg3, (OrganizationDecisionRequest, dict)):
                target_req = arg3
                target_reviewer_id = str(arg4)
            else:
                target_reviewer_id = str(arg3)
                target_req = arg4
        elif target_req is None:
            target_req = arg3 if isinstance(arg3, (OrganizationDecisionRequest, dict)) else arg4
        elif target_reviewer_id is None:
            target_reviewer_id = str(arg3) if not isinstance(arg3, (OrganizationDecisionRequest, dict)) else str(arg4)

        if isinstance(target_req, dict):
            target_req = OrganizationDecisionRequest(**target_req)

        org_id = target_org_id
        reviewer_id = target_reviewer_id
        req = target_req
        reviewer = db.execute(select(User).where(User.id == reviewer_id)).scalar_one_or_none()
        if not reviewer or reviewer.role not in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
            raise ForbiddenException("Only authorized Reviewers or Platform Administrators can make verification decisions.")

        # Conflict of Interest check:
        reviewer_membership = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == reviewer_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        if reviewer_membership:
            raise ForbiddenException("Conflict of Interest: You cannot review or verify an organization you belong to.")

        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        decision = req.decision.upper().strip()
        target_status: OrganizationStatus

        if decision == "APPROVE":
            target_status = OrganizationStatus.VERIFIED
        elif decision == "REJECT":
            target_status = OrganizationStatus.REJECTED
        elif decision in ("REQUEST_REVISION", "REVISION_REQUIRED"):
            target_status = OrganizationStatus.REVISION_REQUIRED
        else:
            raise ValidationException(f"Unsupported decision '{req.decision}'. Allowed: APPROVE, REJECT, REQUEST_REVISION")

        # Validate state transition
        # If currently PENDING, it transitions through UNDER_REVIEW to target
        if org.status == OrganizationStatus.PENDING:
            StateMachineValidator.validate_organization_transition(
                current=org.status,
                target=OrganizationStatus.UNDER_REVIEW,
                actor_platform_role=reviewer.role
            )
            org.status = OrganizationStatus.UNDER_REVIEW

        StateMachineValidator.validate_organization_transition(
            current=org.status,
            target=target_status,
            actor_platform_role=reviewer.role
        )

        now = datetime.utcnow()
        org.status = target_status
        org.reviewer_id = reviewer_id

        if target_status == OrganizationStatus.VERIFIED:
            org.verified_at = now
            org.rejection_reason = None
            org.revision_notes = None
        elif target_status == OrganizationStatus.REJECTED:
            org.rejection_reason = req.reason
        elif target_status == OrganizationStatus.REVISION_REQUIRED:
            org.revision_notes = req.reason

        db.flush()
        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action=f"ORGANIZATION_VERIFICATION_{decision}",
            actor_id=reviewer_id,
            after_state={"status": org.status.value, "decision": decision, "reason": req.reason}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Organization '{org.name}' ({org.id}) verification decision: {decision} by reviewer {reviewer_id}.")
        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def suspend_organization(
        db: Session,
        org_id: str,
        actor_id: str,
        reason: str
    ) -> OrganizationDetailResponse:
        """Suspends an organization and its public capabilities. Requires Platform Admin."""
        user = db.execute(select(User).where(User.id == actor_id)).scalar_one_or_none()
        if not user or user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only Platform Administrators can suspend an organization.")

        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        StateMachineValidator.validate_organization_transition(
            current=org.status,
            target=OrganizationStatus.SUSPENDED,
            actor_platform_role=user.role
        )

        org.status = OrganizationStatus.SUSPENDED
        org.suspension_reason = reason
        db.flush()

        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action="ORGANIZATION_SUSPENDED",
            actor_id=actor_id,
            after_state={"status": org.status.value, "reason": reason}
        )
        db.add(audit)
        db.commit()

        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def reactivate_organization(
        db: Session,
        org_id: str,
        actor_id: str
    ) -> OrganizationDetailResponse:
        """Reactivates a suspended organization to VERIFIED status."""
        user = db.execute(select(User).where(User.id == actor_id)).scalar_one_or_none()
        if not user or user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only Platform Administrators can reactivate an organization.")

        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        StateMachineValidator.validate_organization_transition(
            current=org.status,
            target=OrganizationStatus.VERIFIED,
            actor_platform_role=user.role
        )

        org.status = OrganizationStatus.VERIFIED
        org.suspension_reason = None
        db.flush()

        audit = AuditEvent(
            entity_name="Organization",
            entity_id=org.id,
            action="ORGANIZATION_REACTIVATED",
            actor_id=actor_id,
            after_state={"status": org.status.value}
        )
        db.add(audit)
        db.commit()

        return NGOOnboardingService._build_detail_response(org)

    @staticmethod
    def list_review_queue(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[OrganizationStatus] = None,
        reviewer_id: Optional[str] = None
    ) -> PaginatedResponse[OrganizationDetailResponse]:
        """Lists organizations awaiting review for Reviewer/Admin dashboard."""
        if reviewer_id:
            reviewer = db.execute(select(User).where(User.id == reviewer_id)).scalar_one_or_none()
            if not reviewer or reviewer.role not in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
                raise ForbiddenException("Reviewer or Platform Admin role required.")

        query = select(Organization)
        if status_filter:
            query = query.where(Organization.status == status_filter)
        else:
            query = query.where(
                Organization.status.in_([OrganizationStatus.PENDING, OrganizationStatus.UNDER_REVIEW])
            )

        total_items = db.execute(select(func.count()).select_from(query.subquery())).scalar() or 0
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1

        orgs = db.execute(
            query.order_by(Organization.submitted_at.asc(), Organization.created_at.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).scalars().all()

        return PaginatedResponse(
            data=[NGOOnboardingService._build_detail_response(o) for o in orgs],
            pagination=PaginationMeta(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages
            )
        )

    # --- Helper methods ---

    @staticmethod
    def _get_org_and_verify_membership(
        db: Session,
        org_id: str,
        user_id: str,
        allowed_roles: Tuple[OrganizationRole, ...]
    ) -> Tuple[Organization, OrganizationMembership]:
        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        # Check user platform admin override
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if user and user.role == PlatformRole.PLATFORM_ADMIN:
            # Synthetic membership for admin
            synthetic_mem = OrganizationMembership(
                user_id=user_id,
                organization_id=org_id,
                role=OrganizationRole.OWNER,
                is_active=True
            )
            return org, synthetic_mem

        mem = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == org_id,
                    OrganizationMembership.user_id == user_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        if not mem or mem.role not in allowed_roles:
            raise ForbiddenException("You do not have permission to perform this operation in this organization.")

        return org, mem

    @staticmethod
    def _build_public_response(org: Organization) -> OrganizationPublicResponse:
        focus = json.loads(org.focus_areas) if org.focus_areas else None
        regions = json.loads(org.operating_regions) if org.operating_regions else None
        return OrganizationPublicResponse(
            id=org.id,
            name=org.name,
            display_name=org.display_name or org.name,
            description=org.description,
            organization_type=org.organization_type,
            country=org.country,
            website=org.website,
            mission_statement=org.mission_statement,
            vision=org.vision,
            focus_areas=focus,
            operating_regions=regions,
            years_of_operation=org.years_of_operation or 0,
            status=org.status,
            verified_at=org.verified_at,
            created_at=org.created_at
        )

    @staticmethod
    def _build_detail_response(org: Organization) -> OrganizationDetailResponse:
        focus = json.loads(org.focus_areas) if org.focus_areas else None
        regions = json.loads(org.operating_regions) if org.operating_regions else None
        beneficiaries = json.loads(org.beneficiary_categories) if org.beneficiary_categories else None
        return OrganizationDetailResponse(
            id=org.id,
            name=org.name,
            legal_name=org.legal_name,
            display_name=org.display_name,
            description=org.description,
            organization_type=org.organization_type,
            registration_number=org.registration_number,
            registration_authority=org.registration_authority,
            registration_date=org.registration_date,
            tax_id=org.tax_id,
            classification=org.classification,
            registered_address=org.registered_address,
            operating_address=org.operating_address,
            country=org.country,
            official_email=org.official_email,
            official_phone=org.official_phone,
            website=org.website,
            mission_statement=org.mission_statement,
            vision=org.vision,
            focus_areas=focus,
            operating_regions=regions,
            beneficiary_categories=beneficiaries,
            years_of_operation=org.years_of_operation or 0,
            status=org.status,
            submitted_at=org.submitted_at,
            verified_at=org.verified_at,
            rejection_reason=org.rejection_reason,
            suspension_reason=org.suspension_reason,
            revision_notes=org.revision_notes,
            reviewer_id=getattr(org, "reviewer_id", None),
            reviewed_by_id=getattr(org, "reviewer_id", None),
            created_at=org.created_at
        )

    @staticmethod
    def _build_document_response(doc: OrganizationDocument) -> OrganizationDocumentResponse:
        return OrganizationDocumentResponse(
            id=doc.id,
            organization_id=doc.organization_id,
            document_type=doc.document_type,
            original_filename=doc.original_filename,
            mime_type=doc.mime_type,
            file_size_bytes=doc.file_size_bytes,
            file_hash_sha256=doc.file_hash_sha256,
            version_number=doc.version_number,
            review_status=doc.review_status,
            reviewer_comments=doc.reviewer_comments,
            access_classification=doc.access_classification,
            created_at=doc.created_at
        )

    @staticmethod
    def list_user_organizations(db: Session, user_id: str) -> List[OrganizationDetailResponse]:
        """Lists organizations where the user is an active member."""
        memberships = db.execute(
            select(OrganizationMembership).where(
                and_(OrganizationMembership.user_id == user_id, OrganizationMembership.is_active == True)
            )
        ).scalars().all()
        org_ids = [m.organization_id for m in memberships]
        if not org_ids:
            return []
        orgs = db.execute(
            select(Organization).where(Organization.id.in_(org_ids))
        ).scalars().all()
        return [NGOOnboardingService._build_detail_response(o) for o in orgs]


# Aliases for backward/test compatibility
NGOOnboardingService.get_organization_details = NGOOnboardingService.get_organization
NGOOnboardingService.list_organization_documents = NGOOnboardingService.list_documents
NGOOnboardingService.submit_for_verification = NGOOnboardingService.submit_verification
NGOOnboardingService.get_review_queue = NGOOnboardingService.list_review_queue


