"""
state_machines.py - Domain State Machines & Lifecycle Transition Validation
===========================================================================
Defines the permitted state transitions and authorized actor rules.
Enforces business invariants and prevents arbitrary or unauthorized state changes.
"""

from typing import Dict, Set, Tuple, Optional, Union
from backend.app.core.constants import (
    OrganizationStatus, ProjectStatus, MilestoneStatus,
    EvidenceStatus, PaymentOrderStatus, PaymentAuthorizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus,
    PlatformRole, OrganizationRole, UserRole
)
from backend.app.core.exceptions import InvalidStateTransitionError, UnauthorizedTransitionError


# Permitted state transitions: CurrentState -> Set of Allowed Next States
ORGANIZATION_TRANSITIONS: Dict[OrganizationStatus, Set[OrganizationStatus]] = {
    OrganizationStatus.DRAFT: {OrganizationStatus.PENDING},
    OrganizationStatus.PENDING: {OrganizationStatus.UNDER_REVIEW, OrganizationStatus.VERIFIED, OrganizationStatus.REJECTED, OrganizationStatus.REVISION_REQUIRED, OrganizationStatus.DRAFT},
    OrganizationStatus.UNDER_REVIEW: {OrganizationStatus.VERIFIED, OrganizationStatus.REJECTED, OrganizationStatus.REVISION_REQUIRED},
    OrganizationStatus.REVISION_REQUIRED: {OrganizationStatus.PENDING, OrganizationStatus.DRAFT},
    OrganizationStatus.VERIFIED: {OrganizationStatus.SUSPENDED},
    OrganizationStatus.SUSPENDED: {OrganizationStatus.UNDER_REVIEW, OrganizationStatus.VERIFIED, OrganizationStatus.REJECTED},
    OrganizationStatus.REJECTED: {OrganizationStatus.DRAFT},  # Resubmission allowed as draft
}

PROJECT_TRANSITIONS: Dict[ProjectStatus, Set[ProjectStatus]] = {
    ProjectStatus.DRAFT: {ProjectStatus.SUBMITTED},
    ProjectStatus.SUBMITTED: {ProjectStatus.UNDER_REVIEW, ProjectStatus.APPROVED, ProjectStatus.REJECTED, ProjectStatus.REVISION_REQUIRED, ProjectStatus.DRAFT},
    ProjectStatus.UNDER_REVIEW: {ProjectStatus.APPROVED, ProjectStatus.REVISION_REQUIRED, ProjectStatus.REJECTED},
    ProjectStatus.REVISION_REQUIRED: {ProjectStatus.SUBMITTED, ProjectStatus.DRAFT},
    ProjectStatus.APPROVED: {ProjectStatus.ACTIVE, ProjectStatus.ARCHIVED},
    ProjectStatus.ACTIVE: {ProjectStatus.PAUSED, ProjectStatus.COMPLETED, ProjectStatus.ARCHIVED},
    ProjectStatus.PAUSED: {ProjectStatus.ACTIVE, ProjectStatus.ARCHIVED},
    ProjectStatus.COMPLETED: {ProjectStatus.ARCHIVED},
    ProjectStatus.REJECTED: {ProjectStatus.DRAFT, ProjectStatus.ARCHIVED},
    ProjectStatus.ARCHIVED: set(),  # Terminal state
}

MILESTONE_TRANSITIONS: Dict[MilestoneStatus, Set[MilestoneStatus]] = {
    MilestoneStatus.DRAFT: {MilestoneStatus.SUBMITTED, MilestoneStatus.CANCELLED},
    MilestoneStatus.SUBMITTED: {MilestoneStatus.UNDER_REVIEW, MilestoneStatus.REJECTED, MilestoneStatus.CANCELLED},
    MilestoneStatus.UNDER_REVIEW: {MilestoneStatus.APPROVED, MilestoneStatus.REJECTED, MilestoneStatus.REVISION_REQUIRED},
    MilestoneStatus.REVISION_REQUIRED: {MilestoneStatus.SUBMITTED, MilestoneStatus.CANCELLED},
    MilestoneStatus.APPROVED: {MilestoneStatus.ACTIVE, MilestoneStatus.CANCELLED},
    MilestoneStatus.ACTIVE: {MilestoneStatus.EVIDENCE_SUBMITTED, MilestoneStatus.CANCELLED},
    MilestoneStatus.EVIDENCE_SUBMITTED: {MilestoneStatus.VERIFICATION_PENDING},
    MilestoneStatus.VERIFICATION_PENDING: {MilestoneStatus.COMPLETED, MilestoneStatus.REJECTED, MilestoneStatus.REVISION_REQUIRED},
    MilestoneStatus.COMPLETED: set(),  # Terminal (funds disbursed / certs issued)
    MilestoneStatus.REJECTED: {MilestoneStatus.REVISION_REQUIRED, MilestoneStatus.CANCELLED},
    MilestoneStatus.CANCELLED: set(),  # Terminal state
}

EVIDENCE_TRANSITIONS: Dict[EvidenceStatus, Set[EvidenceStatus]] = {
    EvidenceStatus.UPLOADED: {EvidenceStatus.PROCESSING, EvidenceStatus.FAILED},
    EvidenceStatus.PROCESSING: {EvidenceStatus.PROCESSED, EvidenceStatus.FAILED},
    EvidenceStatus.PROCESSED: {EvidenceStatus.ARCHIVED},
    EvidenceStatus.FAILED: {EvidenceStatus.PROCESSING, EvidenceStatus.ARCHIVED},
    EvidenceStatus.ARCHIVED: set(),
}

PAYMENT_AUTHORIZATION_TRANSITIONS: Dict[PaymentAuthorizationStatus, Set[PaymentAuthorizationStatus]] = {
    PaymentAuthorizationStatus.CREATED: {
        PaymentAuthorizationStatus.CAPTURED,
        PaymentAuthorizationStatus.DENIED,
        PaymentAuthorizationStatus.EXPIRED,
        PaymentAuthorizationStatus.VOIDED
    },
    PaymentAuthorizationStatus.CAPTURED: set(),  # Captured
    PaymentAuthorizationStatus.DENIED: set(),
    PaymentAuthorizationStatus.EXPIRED: set(),
    PaymentAuthorizationStatus.VOIDED: set(),
}

DISBURSEMENT_TRANSITIONS: Dict[DisbursementStatus, Set[DisbursementStatus]] = {
    DisbursementStatus.SCHEDULED: {DisbursementStatus.PROCESSING, DisbursementStatus.CANCELLED},
    DisbursementStatus.PROCESSING: {DisbursementStatus.SUCCEEDED, DisbursementStatus.FAILED},
    DisbursementStatus.SUCCEEDED: set(),  # Terminal successful payout
    DisbursementStatus.FAILED: {DisbursementStatus.SCHEDULED},  # Can be re-scheduled
    DisbursementStatus.CANCELLED: set(),
}


def _resolve_roles(
    actor_platform_role: Optional[PlatformRole] = None,
    actor_org_role: Optional[OrganizationRole] = None,
    actor_role: Optional[Union[UserRole, PlatformRole, OrganizationRole, str]] = None
) -> Tuple[Optional[PlatformRole], Optional[OrganizationRole]]:
    """Resolves platform role and org role with backward compatibility for legacy UserRole."""
    p_role = actor_platform_role
    o_role = actor_org_role

    if actor_role is not None:
        role_str = actor_role.value if hasattr(actor_role, "value") else str(actor_role)
        if role_str == "PLATFORM_ADMIN":
            p_role = PlatformRole.PLATFORM_ADMIN
        elif role_str == "REVIEWER":
            p_role = PlatformRole.REVIEWER
        elif role_str == "DONOR":
            p_role = PlatformRole.DONOR
        elif role_str in ("NGO_ADMIN", "OWNER"):
            o_role = OrganizationRole.OWNER if role_str == "OWNER" else OrganizationRole.ADMIN
        elif role_str in ("NGO_MEMBER", "MEMBER"):
            o_role = OrganizationRole.MEMBER
        elif role_str == "ADMIN":
            o_role = OrganizationRole.ADMIN

    return p_role, o_role


class StateMachineValidator:
    """Validates lifecycle transitions and role permissions."""

    @staticmethod
    def validate_organization_transition(
        current: OrganizationStatus,
        target: OrganizationStatus,
        actor_platform_role: Optional[PlatformRole] = None,
        actor_org_role: Optional[OrganizationRole] = None,
        actor_role: Optional[Union[UserRole, PlatformRole, OrganizationRole, str]] = None
    ):
        """Validates organization onboarding and verification transitions."""
        allowed = ORGANIZATION_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid Organization transition from {current.value} to {target.value}. Allowed: {[s.value for s in allowed]}"
            )

        p_role, o_role = _resolve_roles(actor_platform_role, actor_org_role, actor_role)

        # Organization owner/admin submits application
        if target in {OrganizationStatus.PENDING, OrganizationStatus.DRAFT}:
            is_authorized = (
                p_role == PlatformRole.PLATFORM_ADMIN or
                (o_role in {OrganizationRole.OWNER, OrganizationRole.ADMIN})
            )
            if not is_authorized:
                raise UnauthorizedTransitionError(
                    "Only organization Owners or Admins (or Platform Admins) can submit or draft organization verification."
                )

        # Reviewer / Platform Admin decisions
        if target in {OrganizationStatus.UNDER_REVIEW, OrganizationStatus.VERIFIED, OrganizationStatus.REJECTED, OrganizationStatus.REVISION_REQUIRED, OrganizationStatus.SUSPENDED}:
            if p_role not in {PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Transition to {target.value} requires REVIEWER or PLATFORM_ADMIN platform privileges."
                )

        return True

    @staticmethod
    def validate_milestone_transition(
        current: MilestoneStatus,
        target: MilestoneStatus,
        actor_platform_role: Optional[PlatformRole] = None,
        actor_org_role: Optional[OrganizationRole] = None,
        actor_role: Optional[Union[UserRole, PlatformRole, OrganizationRole, str]] = None
    ):
        allowed = MILESTONE_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid Milestone transition from {current.value} to {target.value}. Allowed: {[s.value for s in allowed]}"
            )
        
        p_role, o_role = _resolve_roles(actor_platform_role, actor_org_role, actor_role)

        # Only REVIEWER or PLATFORM_ADMIN can approve, complete, or request revision after review
        if target in {MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED, MilestoneStatus.REJECTED, MilestoneStatus.REVISION_REQUIRED}:
            if p_role not in {PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Transition to {target.value} requires REVIEWER or PLATFORM_ADMIN."
                )
        
        # Only NGO member/admin/owner or PLATFORM_ADMIN can submit milestone or evidence
        if target in {MilestoneStatus.SUBMITTED, MilestoneStatus.EVIDENCE_SUBMITTED, MilestoneStatus.DRAFT}:
            is_authorized = (
                p_role == PlatformRole.PLATFORM_ADMIN or
                (o_role in {OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER})
            )
            if not is_authorized:
                raise UnauthorizedTransitionError(
                    "Only organization members (or Platform Admins) are authorized to submit milestone or evidence."
                )

        return True

    @staticmethod
    def validate_project_transition(
        current: ProjectStatus,
        target: ProjectStatus,
        actor_platform_role: Optional[PlatformRole] = None,
        actor_org_role: Optional[OrganizationRole] = None,
        actor_role: Optional[Union[UserRole, PlatformRole, OrganizationRole, str]] = None
    ):
        allowed = PROJECT_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid Project transition from {current.value} to {target.value}. Allowed: {[s.value for s in allowed]}"
            )
        
        p_role, o_role = _resolve_roles(actor_platform_role, actor_org_role, actor_role)

        # Reviewer or Platform Admin approves, rejects, requests revision, or moves to under review
        if target in {ProjectStatus.APPROVED, ProjectStatus.REJECTED, ProjectStatus.REVISION_REQUIRED, ProjectStatus.UNDER_REVIEW}:
            if p_role not in {PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Transition to {target.value} requires REVIEWER or PLATFORM_ADMIN."
                )

        # Organization Owner/Admin submits, activates, pauses, or archives
        if target in {ProjectStatus.SUBMITTED, ProjectStatus.ACTIVE, ProjectStatus.PAUSED, ProjectStatus.COMPLETED, ProjectStatus.DRAFT}:
            is_authorized = (
                p_role == PlatformRole.PLATFORM_ADMIN or
                (o_role in {OrganizationRole.OWNER, OrganizationRole.ADMIN})
            )
            if not is_authorized:
                raise UnauthorizedTransitionError(
                    f"Transition to {target.value} requires organization OWNER or ADMIN."
                )

        return True

