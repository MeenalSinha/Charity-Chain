"""
project_service.py - Project, Milestone & Category Domain Service
==================================================================
Manages the complete lifecycle of charitable projects and milestones:
- Database-backed category management with idempotent seeding
- Strict currency-safe budget validation across multiple milestones
- Structured project validation reports
- Role-based lifecycle transitions
- Conflict-of-interest protected reviewer workflows
- Public discovery with database-level filtering and pagination
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func

from backend.app.core.config import settings
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, OrganizationStatus,
    ProjectStatus, MilestoneStatus, ProjectVisibility
)
from backend.app.core.exceptions import (
    ConflictException, NotFoundException, ValidationException, ForbiddenException
)
from backend.app.core.logging import logger
from backend.app.core.state_machines import StateMachineValidator
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.db.models.projects import ProjectCategory, Project, Milestone, ProjectLocation
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.api.schemas import (
    CreateProjectCategoryRequest, UpdateProjectCategoryRequest, ProjectCategoryResponse,
    CreateProjectRequest, UpdateProjectRequest, ProjectResponse, ProjectValidationResponse,
    ProjectValidationIssue, ProjectDecisionRequest,
    CreateMilestoneRequest, UpdateMilestoneRequest, MilestoneResponse,
    PaginationMeta, PaginatedResponse
)


DEFAULT_SEED_CATEGORIES = [
    ("Education", "education", "School construction, educational supplies, teacher training, and scholarships"),
    ("Healthcare", "healthcare", "Clinics, medical camps, maternal health, and disease prevention"),
    ("Clean Water", "clean-water", "Well drilling, water filtration, sanitation facilities, and hygiene"),
    ("Food Security", "food-security", "Emergency food aid, community gardens, nutrition programs, and agriculture"),
    ("Disaster Relief", "disaster-relief", "Emergency response, temporary shelter, search and rescue, and recovery"),
    ("Environmental Protection", "environmental-protection", "Reforestation, ocean cleanup, conservation, and renewable energy"),
    ("Child Welfare", "child-welfare", "Orphan support, foster care, child protection, and nutrition"),
    ("Community Development", "community-development", "Local infrastructure, community centers, and livelihood training"),
    ("Animal Welfare", "animal-welfare", "Animal rescue, veterinary care, wildlife conservation, and shelters"),
    ("Other", "other", "General charitable initiatives not classified elsewhere")
]


class ProjectService:
    """Domain service for Projects, Milestones, and Categories."""

    # ------------------------------------------------------------------------
    # Category Management
    # ------------------------------------------------------------------------

    @staticmethod
    def seed_default_categories(db: Session) -> int:
        """Idempotently seeds standard project categories."""
        seeded = 0
        for name, slug, desc in DEFAULT_SEED_CATEGORIES:
            existing = db.execute(
                select(ProjectCategory).where(
                    or_(ProjectCategory.name == name, ProjectCategory.slug == slug)
                )
            ).scalar_one_or_none()

            if not existing:
                cat = ProjectCategory(name=name, slug=slug, description=desc, is_active=True)
                db.add(cat)
                seeded += 1

        if seeded > 0:
            db.commit()
            logger.info(f"Seeded {seeded} project categories into database.")
        return seeded

    @staticmethod
    def list_categories(db: Session, include_inactive: bool = False) -> List[ProjectCategoryResponse]:
        query = select(ProjectCategory)
        if not include_inactive:
            query = query.where(ProjectCategory.is_active == True)
        cats = db.execute(query.order_by(ProjectCategory.name.asc())).scalars().all()
        return [
            ProjectCategoryResponse(
                id=c.id,
                name=c.name,
                slug=c.slug,
                description=c.description,
                is_active=c.is_active,
                created_at=c.created_at
            )
            for c in cats
        ]

    @staticmethod
    def create_category(db: Session, creator_id: str, req: CreateProjectCategoryRequest) -> ProjectCategoryResponse:
        user = db.execute(select(User).where(User.id == creator_id)).scalar_one_or_none()
        if not user or user.role != PlatformRole.PLATFORM_ADMIN:
            raise ForbiddenException("Only Platform Administrators can create project categories.")

        slug = req.slug or req.name.lower().replace(" ", "-")
        existing = db.execute(
            select(ProjectCategory).where(
                or_(ProjectCategory.name == req.name.strip(), ProjectCategory.slug == slug)
            )
        ).scalar_one_or_none()
        if existing:
            raise ConflictException("Category with this name or slug already exists.")

        cat = ProjectCategory(name=req.name.strip(), slug=slug, description=req.description, is_active=True)
        db.add(cat)
        db.commit()
        return ProjectCategoryResponse(
            id=cat.id, name=cat.name, slug=cat.slug, description=cat.description,
            is_active=cat.is_active, created_at=cat.created_at
        )

    # ------------------------------------------------------------------------
    # Project CRUD & Management
    # ------------------------------------------------------------------------

    @staticmethod
    def create_project(
        db: Session,
        arg2: Any,
        arg3: Any,
        arg4: Any = None
    ) -> ProjectResponse:
        """
        Creates a new draft project for an organization.
        Requires the caller to be an OWNER or ADMIN of the organization.
        """
        if arg4 is not None:
            org_id = str(arg2)
            creator_id = str(arg3)
            req = arg4
        else:
            req = arg2
            creator_id = str(arg3)
            org_id = getattr(req, "organization_id", None) or getattr(req, "org_id", None)
            if not org_id:
                raise ValidationException("organization_id is required to create a project.")
        org, mem = ProjectService._verify_org_membership(
            db, org_id, creator_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        # Validate Currency
        curr = (req.currency or "USD").upper().strip()
        if curr not in settings.SUPPORTED_CURRENCIES:
            raise ValidationException(f"Currency '{curr}' is not supported. Supported: {settings.SUPPORTED_CURRENCIES}")

        # Validate Target Amount
        if req.target_amount_minor_units <= 0:
            raise ValidationException("Project target funding amount must be strictly greater than zero.")

        # Validate Category if provided
        if req.category_id:
            cat = db.execute(select(ProjectCategory).where(ProjectCategory.id == req.category_id)).scalar_one_or_none()
            if not cat or not cat.is_active:
                raise ValidationException("Selected project category is invalid or inactive.")

        # Validate Dates if both provided
        if req.start_date and req.end_date and req.end_date < req.start_date:
            raise ValidationException("Project end date cannot precede start date.")

        # Validate Geolocation if supplied
        ProjectService._validate_coordinates(req.latitude, req.longitude, req.radius_meters)

        project = Project(
            organization_id=org_id,
            category_id=req.category_id,
            title=req.title.strip(),
            summary=req.summary.strip() if req.summary else None,
            description=req.description.strip(),
            objectives=req.objectives,
            project_type=req.project_type or "CHARITABLE_PROJECT",
            public_visibility=req.public_visibility,
            status=ProjectStatus.DRAFT,
            target_amount_minor_units=req.target_amount_minor_units,
            raised_amount_minor_units=0,
            currency=curr,
            target_beneficiaries=req.target_beneficiaries,
            expected_outcomes=req.expected_outcomes,
            deliverables=req.deliverables,
            impact_metrics=req.impact_metrics,
            start_date=req.start_date,
            end_date=req.end_date,
            country=req.country.upper() if req.country else org.country,
            state_region=req.state_region,
            city=req.city,
            address=req.address,
            latitude=req.latitude,
            longitude=req.longitude,
            radius_meters=req.radius_meters or 1000.0,
            creator_id=creator_id,
            version_number=1,
            is_archived=False
        )
        db.add(project)
        db.flush()

        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action="PROJECT_CREATED",
            actor_id=creator_id,
            after_state={"organization_id": org_id, "title": project.title, "target": project.target_amount_minor_units}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Project '{project.title}' ({project.id}) created for org {org_id} by user {creator_id}.")
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def get_project(db: Session, project_id: str, requesting_user_id: Optional[str] = None) -> ProjectResponse:
        """Retrieves project details enforcing visibility access control."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        # If project is active and public, everyone can view
        if project.status == ProjectStatus.ACTIVE and project.public_visibility == ProjectVisibility.PUBLIC:
            return ProjectService._build_project_response(project, include_milestones=True)

        # Non-public / draft / review projects require authorization
        if not requesting_user_id:
            raise ForbiddenException("This project is not publicly visible.")

        user = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
        is_reviewer_or_admin = user and user.role in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN)

        if not is_reviewer_or_admin:
            mem = db.execute(
                select(OrganizationMembership).where(
                    and_(
                        OrganizationMembership.organization_id == project.organization_id,
                        OrganizationMembership.user_id == requesting_user_id,
                        OrganizationMembership.is_active == True
                    )
                )
            ).scalar_one_or_none()
            if not mem:
                raise ForbiddenException("You do not have permission to view this non-public project.")

        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def update_project(
        db: Session,
        project_id: str,
        user_id: str,
        req: UpdateProjectRequest
    ) -> ProjectResponse:
        """Updates project details. Permitted when in DRAFT or REVISION_REQUIRED."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        if project.status not in (ProjectStatus.DRAFT, ProjectStatus.REVISION_REQUIRED, ProjectStatus.SUBMITTED):
            raise ValidationException(f"Cannot edit project while in status {project.status.value}.")

        if req.title is not None:
            project.title = req.title.strip()
        if req.summary is not None:
            project.summary = req.summary.strip()
        if req.description is not None:
            project.description = req.description.strip()
        if req.category_id is not None:
            cat = db.execute(select(ProjectCategory).where(ProjectCategory.id == req.category_id)).scalar_one_or_none()
            if not cat or not cat.is_active:
                raise ValidationException("Selected category is invalid.")
            project.category_id = req.category_id
        if req.objectives is not None:
            project.objectives = req.objectives
        if req.public_visibility is not None:
            project.public_visibility = req.public_visibility
        if req.target_amount_minor_units is not None:
            if req.target_amount_minor_units <= 0:
                raise ValidationException("Project funding target must be greater than zero.")
            # Check milestone sum does not exceed new target
            current_milestone_sum = sum(m.target_amount_minor_units for m in project.milestones)
            if current_milestone_sum > req.target_amount_minor_units:
                raise ValidationException(
                    f"New target ({req.target_amount_minor_units}) is less than current milestone allocations ({current_milestone_sum})."
                )
            project.target_amount_minor_units = req.target_amount_minor_units
        if req.currency is not None:
            curr = req.currency.upper().strip()
            if curr not in settings.SUPPORTED_CURRENCIES:
                raise ValidationException(f"Unsupported currency '{curr}'.")
            project.currency = curr
        if req.target_beneficiaries is not None:
            project.target_beneficiaries = req.target_beneficiaries
        if req.expected_outcomes is not None:
            project.expected_outcomes = req.expected_outcomes
        if req.deliverables is not None:
            project.deliverables = req.deliverables
        if req.impact_metrics is not None:
            project.impact_metrics = req.impact_metrics
        if req.start_date is not None:
            project.start_date = req.start_date
        if req.end_date is not None:
            project.end_date = req.end_date
        if project.start_date and project.end_date and project.end_date < project.start_date:
            raise ValidationException("End date cannot precede start date.")

        if req.country is not None:
            project.country = req.country.upper()
        if req.state_region is not None:
            project.state_region = req.state_region
        if req.city is not None:
            project.city = req.city
        if req.address is not None:
            project.address = req.address
        if req.latitude is not None or req.longitude is not None:
            lat = req.latitude if req.latitude is not None else project.latitude
            lon = req.longitude if req.longitude is not None else project.longitude
            rad = req.radius_meters if req.radius_meters is not None else project.radius_meters
            ProjectService._validate_coordinates(lat, lon, rad)
            project.latitude = lat
            project.longitude = lon
            project.radius_meters = rad

        project.version_number += 1
        db.flush()

        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action="PROJECT_UPDATED",
            actor_id=user_id,
            after_state={"title": project.title, "version": project.version_number}
        )
        db.add(audit)
        db.commit()

        return ProjectService._build_project_response(project, include_milestones=True)

    # ------------------------------------------------------------------------
    # Project Submission & Structured Validation
    # ------------------------------------------------------------------------

    @staticmethod
    def validate_project_submission(db: Session, project_id: str, user_id: Optional[str] = None) -> ProjectValidationResponse:
        """
        Performs exhaustive business rule checks on project prior to submission.
        Returns a structured validation report with all discovered issues.
        """
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        issues: List[ProjectValidationIssue] = []

        # 1. Organization Check
        org = project.organization
        if not org:
            issues.append(ProjectValidationIssue(field="organization", code="ORG_NOT_FOUND", message="Organization not found."))
        else:
            if org.status != OrganizationStatus.VERIFIED:
                issues.append(ProjectValidationIssue(
                    field="organization",
                    code="ORG_NOT_VERIFIED",
                    message=f"Organization must be VERIFIED before submitting projects (current status: {org.status.value})."
                ))
            if org.status == OrganizationStatus.SUSPENDED:
                issues.append(ProjectValidationIssue(
                    field="organization",
                    code="ORG_SUSPENDED",
                    message="Suspended organizations cannot submit projects."
                ))

        # 2. Content Checks
        if not project.title or len(project.title.strip()) < 3:
            issues.append(ProjectValidationIssue(field="title", code="TITLE_TOO_SHORT", message="Project title must be at least 3 characters."))
        if not project.description or len(project.description.strip()) < 10:
            issues.append(ProjectValidationIssue(field="description", code="DESCRIPTION_TOO_SHORT", message="Project description must be at least 10 characters."))

        # 3. Financial Checks
        if project.target_amount_minor_units <= 0:
            issues.append(ProjectValidationIssue(field="target_amount", code="INVALID_TARGET", message="Funding target must be greater than zero."))
        if project.currency not in settings.SUPPORTED_CURRENCIES:
            issues.append(ProjectValidationIssue(field="currency", code="UNSUPPORTED_CURRENCY", message=f"Currency '{project.currency}' is unsupported."))

        # 4. Milestone Checks
        milestones = project.milestones
        if not milestones:
            issues.append(ProjectValidationIssue(field="milestones", code="MILESTONES_REQUIRED", message="At least one milestone is required prior to project submission."))
        else:
            milestone_total = sum(m.target_amount_minor_units for m in milestones)
            if milestone_total > project.target_amount_minor_units:
                issues.append(ProjectValidationIssue(
                    field="milestone_budget",
                    code="OVER_ALLOCATED",
                    message=f"Sum of milestone allocations ({milestone_total}) exceeds project funding target ({project.target_amount_minor_units})."
                ))
            for m in milestones:
                if m.currency != project.currency:
                    issues.append(ProjectValidationIssue(
                        field="milestone_currency",
                        code="CURRENCY_MISMATCH",
                        message=f"Milestone '{m.title}' currency ({m.currency}) does not match project currency ({project.currency})."
                    ))
                if m.target_amount_minor_units < 0:
                    issues.append(ProjectValidationIssue(
                        field="milestone_target",
                        code="NEGATIVE_ALLOCATION",
                        message=f"Milestone '{m.title}' target amount cannot be negative."
                    ))

        # 5. Timeline Checks
        if project.start_date and project.end_date and project.end_date < project.start_date:
            issues.append(ProjectValidationIssue(field="dates", code="INVALID_TIMELINE", message="Project end date cannot precede start date."))

        is_valid = len(issues) == 0
        return ProjectValidationResponse(valid=is_valid, can_submit=is_valid, issues=issues)

    @staticmethod
    def submit_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        """Validates and submits project for reviewer inspection."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        val_report = ProjectService.validate_project_submission(db, project_id)
        if not val_report.valid:
            issue_messages = "; ".join([f"[{i.field}] {i.message}" for i in val_report.issues])
            raise ValidationException(f"Project submission validation failed: {issue_messages}")

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.SUBMITTED,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.SUBMITTED
        project.submitted_at = datetime.utcnow()
        db.flush()

        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action="PROJECT_SUBMITTED",
            actor_id=user_id,
            after_state={"status": project.status.value, "submitted_at": project.submitted_at.isoformat()}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Project '{project.title}' ({project.id}) submitted for review by {user_id}.")
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def withdraw_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        """Withdraws a submitted project back to DRAFT."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.DRAFT,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.DRAFT
        db.flush()

        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action="PROJECT_WITHDRAWN",
            actor_id=user_id,
            after_state={"status": project.status.value}
        )
        db.add(audit)
        db.commit()

        return ProjectService._build_project_response(project, include_milestones=True)

    # ------------------------------------------------------------------------
    # Reviewer Decisions & Conflict of Interest Check
    # ------------------------------------------------------------------------

    @staticmethod
    def review_project(
        db: Session,
        project_id: Optional[str] = None,
        arg3: Any = None,
        arg4: Any = None,
        decision: Any = None,
        reviewer_id: Optional[str] = None,
        req: Any = None
    ) -> ProjectResponse:
        """
        Executes a reviewer decision on a submitted project.
        Enforces Conflict-of-Interest prevention: Reviewers cannot review projects
        from organizations they belong to.
        """
        target_req = req or decision
        target_reviewer_id = reviewer_id

        if target_req is None and target_reviewer_id is None:
            if isinstance(arg3, (ProjectDecisionRequest, dict)):
                target_req = arg3
                target_reviewer_id = str(arg4)
            else:
                target_reviewer_id = str(arg3)
                target_req = arg4
        elif target_req is None:
            target_req = arg3 if isinstance(arg3, (ProjectDecisionRequest, dict)) else arg4
        elif target_reviewer_id is None:
            target_reviewer_id = str(arg3) if not isinstance(arg3, (ProjectDecisionRequest, dict)) else str(arg4)

        if isinstance(target_req, dict):
            target_req = ProjectDecisionRequest(**target_req)

        reviewer_id = target_reviewer_id
        req = target_req
        reviewer = db.execute(select(User).where(User.id == reviewer_id)).scalar_one_or_none()
        if not reviewer or reviewer.role not in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
            raise ForbiddenException("Only authorized Reviewers or Platform Administrators can review projects.")

        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        # Conflict-of-interest check
        is_member = db.execute(
            select(OrganizationMembership).where(
                and_(
                    OrganizationMembership.organization_id == project.organization_id,
                    OrganizationMembership.user_id == reviewer_id,
                    OrganizationMembership.is_active == True
                )
            )
        ).scalar_one_or_none()

        if is_member:
            raise ForbiddenException("Conflict of Interest: You cannot review a project from an organization you belong to.")

        decision = req.decision.upper().strip()
        target_status: ProjectStatus

        if decision == "APPROVE":
            target_status = ProjectStatus.APPROVED
        elif decision == "REJECT":
            target_status = ProjectStatus.REJECTED
        elif decision in ("REQUEST_REVISION", "REVISION_REQUIRED"):
            target_status = ProjectStatus.REVISION_REQUIRED
        else:
            raise ValidationException(f"Unsupported decision '{req.decision}'. Allowed: APPROVE, REJECT, REQUEST_REVISION")

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=target_status,
            actor_platform_role=reviewer.role
        )

        now = datetime.utcnow()
        project.status = target_status
        project.reviewer_id = reviewer_id
        project.reviewed_at = now

        if target_status == ProjectStatus.APPROVED:
            project.rejection_reason = None
            project.revision_notes = None
        elif target_status == ProjectStatus.REJECTED:
            project.rejection_reason = req.reason
        elif target_status == ProjectStatus.REVISION_REQUIRED:
            project.revision_notes = req.reason

        db.flush()
        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action=f"PROJECT_REVIEW_{decision}",
            actor_id=reviewer_id,
            after_state={"status": project.status.value, "decision": decision, "reason": req.reason}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Project '{project.title}' ({project.id}) reviewed: {decision} by reviewer {reviewer_id}.")
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def activate_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        """Activates an approved project, making it publicly discoverable."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.ACTIVE,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.ACTIVE
        project.activated_at = datetime.utcnow()
        db.flush()

        audit = AuditEvent(
            entity_name="Project",
            entity_id=project.id,
            action="PROJECT_ACTIVATED",
            actor_id=user_id,
            after_state={"status": project.status.value, "activated_at": project.activated_at.isoformat()}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Project '{project.title}' ({project.id}) activated by {user_id}.")
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def pause_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.PAUSED,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.PAUSED
        db.flush()

        audit = AuditEvent(entity_name="Project", entity_id=project.id, action="PROJECT_PAUSED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def resume_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.ACTIVE,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.ACTIVE
        db.flush()

        audit = AuditEvent(entity_name="Project", entity_id=project.id, action="PROJECT_RESUMED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def complete_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.COMPLETED,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.COMPLETED
        db.flush()

        audit = AuditEvent(entity_name="Project", entity_id=project.id, action="PROJECT_COMPLETED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return ProjectService._build_project_response(project, include_milestones=True)

    @staticmethod
    def archive_project(db: Session, project_id: str, user_id: str) -> ProjectResponse:
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        org, mem = ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        StateMachineValidator.validate_project_transition(
            current=project.status,
            target=ProjectStatus.ARCHIVED,
            actor_org_role=mem.role
        )

        project.status = ProjectStatus.ARCHIVED
        project.is_archived = True
        db.flush()

        audit = AuditEvent(entity_name="Project", entity_id=project.id, action="PROJECT_ARCHIVED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return ProjectService._build_project_response(project, include_milestones=True)

    # ------------------------------------------------------------------------
    # Multi-Milestone Management
    # ------------------------------------------------------------------------

    @staticmethod
    def create_milestone(
        db: Session,
        project_id: str,
        arg3: Any,
        arg4: Any
    ) -> MilestoneResponse:
        """Adds a milestone to a project enforcing sequence and budget rules."""
        if isinstance(arg3, CreateMilestoneRequest):
            req = arg3
            user_id = str(arg4)
        else:
            user_id = str(arg3)
            req = arg4
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")

        ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        if project.status not in (ProjectStatus.DRAFT, ProjectStatus.REVISION_REQUIRED):
            raise ValidationException(f"Cannot add milestones while project is in status {project.status.value}.")

        # Check Currency
        milestone_curr = (req.currency or project.currency).upper().strip()
        if milestone_curr != project.currency:
            raise ValidationException(f"Milestone currency ({milestone_curr}) must match project currency ({project.currency}).")

        # Determine sequence number
        existing_milestones = project.milestones
        if req.sequence_number is not None:
            seq = req.sequence_number
            if any(m.sequence_number == seq for m in existing_milestones):
                raise ConflictException(f"A milestone with sequence number {seq} already exists in this project.")
        else:
            seq = max([m.sequence_number for m in existing_milestones], default=0) + 1

        # Check budget allocation
        current_sum = sum(m.target_amount_minor_units for m in existing_milestones)
        if current_sum + req.target_amount_minor_units > project.target_amount_minor_units:
            raise ValidationException(
                f"Adding milestone with target {req.target_amount_minor_units} causes total milestone allocations "
                f"({current_sum + req.target_amount_minor_units}) to exceed project funding target ({project.target_amount_minor_units})."
            )

        ProjectService._validate_coordinates(req.latitude, req.longitude, req.radius_meters)

        milestone = Milestone(
            project_id=project_id,
            title=req.title.strip(),
            description=req.description.strip(),
            objective=req.objective,
            deliverables=req.deliverables,
            expected_outcome=req.expected_outcome,
            success_criteria=req.success_criteria,
            required_evidence_desc=req.required_evidence_desc,
            sequence_number=seq,
            target_amount_minor_units=req.target_amount_minor_units,
            raised_amount_minor_units=0,
            currency=milestone_curr,
            status=MilestoneStatus.DRAFT,
            planned_start_date=req.planned_start_date,
            planned_completion_date=req.planned_completion_date,
            latitude=req.latitude if req.latitude is not None else project.latitude,
            longitude=req.longitude if req.longitude is not None else project.longitude,
            radius_meters=req.radius_meters or 1000.0,
            address=req.address or project.address,
            city=req.city or project.city,
            country=req.country or project.country,
            version_number=1
        )
        db.add(milestone)
        db.flush()

        # Also populate ProjectLocation for backward compatibility
        if milestone.latitude is not None and milestone.longitude is not None:
            ploc = ProjectLocation(
                milestone_id=milestone.id,
                latitude=milestone.latitude,
                longitude=milestone.longitude,
                address=milestone.address,
                city=milestone.city,
                country=milestone.country,
                radius_meters=milestone.radius_meters
            )
            db.add(ploc)

        audit = AuditEvent(
            entity_name="Milestone",
            entity_id=milestone.id,
            action="MILESTONE_CREATED",
            actor_id=user_id,
            after_state={"project_id": project_id, "title": milestone.title, "seq": seq}
        )
        db.add(audit)
        db.commit()

        logger.info(f"Milestone '{milestone.title}' (#{seq}) created for project {project_id}.")
        return ProjectService._build_milestone_response(milestone)

    @staticmethod
    def update_milestone(
        db: Session,
        milestone_id: str,
        user_id: str,
        req: UpdateMilestoneRequest
    ) -> MilestoneResponse:
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException("Milestone not found.")

        project = milestone.project
        ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        if project.status not in (ProjectStatus.DRAFT, ProjectStatus.REVISION_REQUIRED):
            raise ValidationException(f"Cannot edit milestone while project is in status {project.status.value}.")

        if req.title is not None:
            milestone.title = req.title.strip()
        if req.description is not None:
            milestone.description = req.description.strip()
        if req.target_amount_minor_units is not None:
            other_sum = sum(m.target_amount_minor_units for m in project.milestones if m.id != milestone.id)
            if other_sum + req.target_amount_minor_units > project.target_amount_minor_units:
                raise ValidationException(
                    f"New milestone amount causes total allocations ({other_sum + req.target_amount_minor_units}) "
                    f"to exceed project target ({project.target_amount_minor_units})."
                )
            milestone.target_amount_minor_units = req.target_amount_minor_units
        if req.objective is not None:
            milestone.objective = req.objective
        if req.deliverables is not None:
            milestone.deliverables = req.deliverables
        if req.expected_outcome is not None:
            milestone.expected_outcome = req.expected_outcome
        if req.success_criteria is not None:
            milestone.success_criteria = req.success_criteria
        if req.required_evidence_desc is not None:
            milestone.required_evidence_desc = req.required_evidence_desc
        if req.planned_start_date is not None:
            milestone.planned_start_date = req.planned_start_date
        if req.planned_completion_date is not None:
            milestone.planned_completion_date = req.planned_completion_date
        if req.latitude is not None or req.longitude is not None:
            lat = req.latitude if req.latitude is not None else milestone.latitude
            lon = req.longitude if req.longitude is not None else milestone.longitude
            rad = req.radius_meters if req.radius_meters is not None else milestone.radius_meters
            ProjectService._validate_coordinates(lat, lon, rad)
            milestone.latitude = lat
            milestone.longitude = lon
            milestone.radius_meters = rad

        milestone.version_number += 1
        db.flush()

        audit = AuditEvent(entity_name="Milestone", entity_id=milestone.id, action="MILESTONE_UPDATED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return ProjectService._build_milestone_response(milestone)

    @staticmethod
    def delete_milestone(db: Session, milestone_id: str, user_id: str) -> bool:
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException("Milestone not found.")

        project = milestone.project
        ProjectService._verify_org_membership(
            db, project.organization_id, user_id, allowed_roles=(OrganizationRole.OWNER, OrganizationRole.ADMIN)
        )

        if project.status not in (ProjectStatus.DRAFT, ProjectStatus.REVISION_REQUIRED):
            raise ValidationException("Cannot delete milestones after project is submitted or approved.")

        # Check for donations or evidence
        if milestone.donations:
            raise ValidationException("Cannot delete a milestone that has recorded donations.")

        project_id = project.id
        db.delete(milestone)
        db.flush()

        # Re-index remaining milestones sequence numbers
        remaining = db.execute(
            select(Milestone).where(Milestone.project_id == project_id).order_by(Milestone.sequence_number.asc())
        ).scalars().all()
        for i, m in enumerate(remaining, start=1):
            m.sequence_number = i

        audit = AuditEvent(entity_name="Milestone", entity_id=milestone_id, action="MILESTONE_DELETED", actor_id=user_id)
        db.add(audit)
        db.commit()
        return True

    @staticmethod
    def list_milestones(db: Session, project_id: str) -> List[MilestoneResponse]:
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project or project.is_archived:
            raise NotFoundException("Project not found.")
        milestones = db.execute(
            select(Milestone).where(Milestone.project_id == project_id).order_by(Milestone.sequence_number.asc())
        ).scalars().all()
        return [ProjectService._build_milestone_response(m) for m in milestones]

    # ------------------------------------------------------------------------
    # Public Project Discovery & Filters
    # ------------------------------------------------------------------------

    @staticmethod
    def discover_projects(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        category_id: Optional[str] = None,
        country: Optional[str] = None,
        min_target: Optional[int] = None,
        max_target: Optional[int] = None
    ) -> PaginatedResponse[ProjectResponse]:
        """
        Public discovery query.
        Filters strictly for:
        1. ProjectStatus.ACTIVE (or COMPLETED)
        2. public_visibility == PUBLIC
        3. OrganizationStatus == VERIFIED
        4. is_archived == False
        """
        query = (
            select(Project)
            .join(Organization, Project.organization_id == Organization.id)
            .where(
                and_(
                    Project.status.in_([ProjectStatus.ACTIVE, ProjectStatus.COMPLETED]),
                    Project.public_visibility == ProjectVisibility.PUBLIC,
                    Project.is_archived == False,
                    Organization.status == OrganizationStatus.VERIFIED
                )
            )
        )

        if search:
            s = f"%{search.strip().lower()}%"
            query = query.where(
                or_(
                    func.lower(Project.title).like(s),
                    func.lower(Project.description).like(s),
                    func.lower(Project.summary).like(s)
                )
            )

        if category_id:
            query = query.where(Project.category_id == category_id)

        if country:
            query = query.where(Project.country == country.upper().strip())

        if min_target is not None:
            query = query.where(Project.target_amount_minor_units >= min_target)

        if max_target is not None:
            query = query.where(Project.target_amount_minor_units <= max_target)

        total_items = db.execute(select(func.count()).select_from(query.subquery())).scalar() or 0
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1

        projects = db.execute(
            query.order_by(Project.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).scalars().all()

        return PaginatedResponse(
            data=[ProjectService._build_project_response(p, include_milestones=True) for p in projects],
            pagination=PaginationMeta(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages
            )
        )

    @staticmethod
    def list_organization_projects(
        db: Session,
        org_id: str,
        requesting_user_id: Optional[str] = None
    ) -> List[ProjectResponse]:
        """Lists projects for an organization. Shows draft/unapproved only to members or admins."""
        is_member_or_admin = False
        if requesting_user_id:
            user = db.execute(select(User).where(User.id == requesting_user_id)).scalar_one_or_none()
            if user and user.role in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
                is_member_or_admin = True
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
                if mem:
                    is_member_or_admin = True

        query = select(Project).where(
            and_(
                Project.organization_id == org_id,
                Project.is_archived == False
            )
        )
        if not is_member_or_admin:
            query = query.where(
                and_(
                    Project.status == ProjectStatus.ACTIVE,
                    Project.public_visibility == ProjectVisibility.PUBLIC
                )
            )

        projects = db.execute(query.order_by(Project.created_at.desc())).scalars().all()
        return [ProjectService._build_project_response(p, include_milestones=True) for p in projects]

    @staticmethod
    def list_review_queue(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        reviewer_id: Optional[str] = None
    ) -> PaginatedResponse[ProjectResponse]:
        """Lists submitted projects awaiting reviewer approval."""
        if reviewer_id:
            reviewer = db.execute(select(User).where(User.id == reviewer_id)).scalar_one_or_none()
            if not reviewer or reviewer.role not in (PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN):
                raise ForbiddenException("Only Reviewers or Platform Administrators can access review queues.")

        query = select(Project).where(
            and_(
                Project.status.in_([ProjectStatus.SUBMITTED, ProjectStatus.UNDER_REVIEW]),
                Project.is_archived == False
            )
        )

        total_items = db.execute(select(func.count()).select_from(query.subquery())).scalar() or 0
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1

        projects = db.execute(
            query.order_by(Project.submitted_at.asc(), Project.created_at.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).scalars().all()

        return PaginatedResponse(
            data=[ProjectService._build_project_response(p, include_milestones=True) for p in projects],
            pagination=PaginationMeta(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages
            )
        )

    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _validate_coordinates(lat: Optional[float], lon: Optional[float], radius: Optional[float]):
        if lat is not None:
            if not (-90.0 <= lat <= 90.0):
                raise ValidationException(f"Latitude ({lat}) must be between -90.0 and 90.0.")
        if lon is not None:
            if not (-180.0 <= lon <= 180.0):
                raise ValidationException(f"Longitude ({lon}) must be between -180.0 and 180.0.")
        if radius is not None and radius <= 0:
            raise ValidationException("Geofence radius must be strictly positive.")

    @staticmethod
    def _verify_org_membership(
        db: Session,
        org_id: str,
        user_id: str,
        allowed_roles: Tuple[OrganizationRole, ...]
    ) -> Tuple[Organization, OrganizationMembership]:
        org = db.execute(select(Organization).where(Organization.id == org_id)).scalar_one_or_none()
        if not org:
            raise NotFoundException("Organization not found.")

        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if user and user.role == PlatformRole.PLATFORM_ADMIN:
            synthetic_mem = OrganizationMembership(
                user_id=user_id, organization_id=org_id, role=OrganizationRole.OWNER, is_active=True
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
            raise ForbiddenException("You do not have permission to manage projects for this organization.")

        return org, mem

    @staticmethod
    def _build_milestone_response(m: Milestone) -> MilestoneResponse:
        return MilestoneResponse(
            id=m.id,
            project_id=m.project_id,
            title=m.title,
            description=m.description,
            sequence_number=m.sequence_number,
            target_amount_minor_units=m.target_amount_minor_units,
            raised_amount_minor_units=m.raised_amount_minor_units,
            currency=m.currency,
            status=m.status,
            objective=m.objective,
            deliverables=m.deliverables,
            expected_outcome=m.expected_outcome,
            success_criteria=m.success_criteria,
            required_evidence_desc=m.required_evidence_desc,
            planned_start_date=m.planned_start_date,
            planned_completion_date=m.planned_completion_date,
            latitude=m.latitude,
            longitude=m.longitude,
            radius_meters=m.radius_meters,
            address=m.address,
            city=m.city,
            country=m.country,
            created_at=m.created_at
        )

    @staticmethod
    def _build_project_response(p: Project, include_milestones: bool = False) -> ProjectResponse:
        milestones = [ProjectService._build_milestone_response(m) for m in p.milestones] if include_milestones else None
        return ProjectResponse(
            id=p.id,
            organization_id=p.organization_id,
            organization_name=p.organization.name if p.organization else None,
            category_id=p.category_id,
            category_name=p.category.name if p.category else None,
            title=p.title,
            summary=p.summary,
            description=p.description,
            objectives=p.objectives,
            project_type=p.project_type,
            public_visibility=p.public_visibility,
            status=p.status,
            target_amount_minor_units=p.target_amount_minor_units,
            raised_amount_minor_units=p.raised_amount_minor_units,
            currency=p.currency,
            target_beneficiaries=p.target_beneficiaries,
            expected_outcomes=p.expected_outcomes,
            deliverables=p.deliverables,
            impact_metrics=p.impact_metrics,
            start_date=p.start_date,
            end_date=p.end_date,
            country=p.country,
            state_region=p.state_region,
            city=p.city,
            address=p.address,
            latitude=p.latitude,
            longitude=p.longitude,
            radius_meters=p.radius_meters,
            submitted_at=p.submitted_at,
            reviewed_at=p.reviewed_at,
            activated_at=p.activated_at,
            rejection_reason=p.rejection_reason,
            revision_notes=p.revision_notes,
            milestone_count=len(p.milestones) if p.milestones else 0,
            milestones=milestones,
            created_at=p.created_at
        )


# Aliases for backward/test compatibility
ProjectService.get_project_details = ProjectService.get_project
ProjectService.list_project_milestones = ProjectService.list_milestones
ProjectService.list_public_projects = ProjectService.discover_projects
ProjectService.get_review_queue = ProjectService.list_review_queue

