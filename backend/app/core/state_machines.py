"""
state_machines.py - Domain State Machines & Lifecycle Transition Validation
===========================================================================
Defines the permitted state transitions and authorized actor rules.
Enforces business invariants and prevents arbitrary or unauthorized state changes.
"""

from typing import Dict, Set, Tuple
from backend.app.core.constants import (
    OrganizationStatus, ProjectStatus, MilestoneStatus,
    EvidenceStatus, PaymentOrderStatus, PaymentAuthorizationStatus,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus,
    UserRole
)
from backend.app.core.exceptions import InvalidStateTransitionError, UnauthorizedTransitionError


# Permitted state transitions: CurrentState -> Set of Allowed Next States
ORGANIZATION_TRANSITIONS: Dict[OrganizationStatus, Set[OrganizationStatus]] = {
    OrganizationStatus.PENDING: {OrganizationStatus.UNDER_REVIEW, OrganizationStatus.REJECTED},
    OrganizationStatus.UNDER_REVIEW: {OrganizationStatus.VERIFIED, OrganizationStatus.REJECTED},
    OrganizationStatus.VERIFIED: {OrganizationStatus.SUSPENDED},
    OrganizationStatus.SUSPENDED: {OrganizationStatus.UNDER_REVIEW, OrganizationStatus.REJECTED},
    OrganizationStatus.REJECTED: set(),  # Terminal state
}

PROJECT_TRANSITIONS: Dict[ProjectStatus, Set[ProjectStatus]] = {
    ProjectStatus.DRAFT: {ProjectStatus.SUBMITTED},
    ProjectStatus.SUBMITTED: {ProjectStatus.APPROVED, ProjectStatus.REJECTED},
    ProjectStatus.APPROVED: {ProjectStatus.ACTIVE, ProjectStatus.ARCHIVED},
    ProjectStatus.ACTIVE: {ProjectStatus.PAUSED, ProjectStatus.COMPLETED},
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


class StateMachineValidator:
    """Validates lifecycle transitions and role permissions."""

    @staticmethod
    def validate_milestone_transition(
        current: MilestoneStatus,
        target: MilestoneStatus,
        actor_role: UserRole
    ):
        allowed = MILESTONE_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid Milestone transition from {current.value} to {target.value}. Allowed: {[s.value for s in allowed]}"
            )
        
        # Role-based transition permissions:
        # Only REVIEWER or PLATFORM_ADMIN can approve, complete, or request revision after review
        if target in {MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED, MilestoneStatus.REJECTED, MilestoneStatus.REVISION_REQUIRED}:
            if actor_role not in {UserRole.REVIEWER, UserRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Role {actor_role.value} is not authorized to transition Milestone to {target.value}. Requires REVIEWER or PLATFORM_ADMIN."
                )
        
        # Only NGO_ADMIN or PLATFORM_ADMIN can submit or draft
        if target in {MilestoneStatus.SUBMITTED, MilestoneStatus.EVIDENCE_SUBMITTED}:
            if actor_role not in {UserRole.NGO_ADMIN, UserRole.NGO_MEMBER, UserRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Role {actor_role.value} is not authorized to submit milestone or evidence."
                )

    @staticmethod
    def validate_project_transition(
        current: ProjectStatus,
        target: ProjectStatus,
        actor_role: UserRole
    ):
        allowed = PROJECT_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid Project transition from {current.value} to {target.value}. Allowed: {[s.value for s in allowed]}"
            )
        if target in {ProjectStatus.APPROVED, ProjectStatus.REJECTED}:
            if actor_role not in {UserRole.REVIEWER, UserRole.PLATFORM_ADMIN}:
                raise UnauthorizedTransitionError(
                    f"Role {actor_role.value} cannot approve/reject projects."
                )

