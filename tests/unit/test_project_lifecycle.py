"""
test_project_lifecycle.py - Unit tests for Project State Machine and Lifecycle
=============================================================================
Verifies project state transitions, role authorizations, withdrawal flows,
and state machine invariants.
"""

import pytest
from backend.app.core.constants import ProjectStatus, PlatformRole, OrganizationRole
from backend.app.core.state_machines import StateMachineValidator
from backend.app.core.exceptions import InvalidStateTransitionError, UnauthorizedTransitionError


def test_project_status_valid_transitions():
    """Verify legitimate project lifecycle paths."""
    # DRAFT -> SUBMITTED (Actor: Organization Admin/Owner)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.DRAFT,
        ProjectStatus.SUBMITTED,
        actor_org_role=OrganizationRole.OWNER
    )

    # SUBMITTED -> DRAFT (Withdrawal by submitter)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.SUBMITTED,
        ProjectStatus.DRAFT,
        actor_org_role=OrganizationRole.ADMIN
    )

    # SUBMITTED -> APPROVED (Actor: Reviewer)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.SUBMITTED,
        ProjectStatus.APPROVED,
        actor_platform_role=PlatformRole.REVIEWER
    )

    # SUBMITTED -> REVISION_REQUIRED (Actor: Reviewer)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.SUBMITTED,
        ProjectStatus.REVISION_REQUIRED,
        actor_platform_role=PlatformRole.REVIEWER
    )

    # REVISION_REQUIRED -> SUBMITTED (Actor: Owner resubmits)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.REVISION_REQUIRED,
        ProjectStatus.SUBMITTED,
        actor_org_role=OrganizationRole.OWNER
    )

    # APPROVED -> ACTIVE (Actor: Organization Owner activates project)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.APPROVED,
        ProjectStatus.ACTIVE,
        actor_org_role=OrganizationRole.OWNER
    )

    # ACTIVE -> PAUSED (Actor: Organization Admin or Platform Admin)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.ACTIVE,
        ProjectStatus.PAUSED,
        actor_org_role=OrganizationRole.ADMIN
    )

    # PAUSED -> ACTIVE (Actor: Organization Admin)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.PAUSED,
        ProjectStatus.ACTIVE,
        actor_org_role=OrganizationRole.ADMIN
    )

    # ACTIVE -> COMPLETED (Actor: Organization Owner or Platform Admin)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.ACTIVE,
        ProjectStatus.COMPLETED,
        actor_org_role=OrganizationRole.OWNER
    )

    # COMPLETED -> ARCHIVED (Actor: Platform Admin)
    assert StateMachineValidator.validate_project_transition(
        ProjectStatus.COMPLETED,
        ProjectStatus.ARCHIVED,
        actor_platform_role=PlatformRole.PLATFORM_ADMIN
    )


def test_project_status_invalid_transitions():
    """Verify illegal transitions are rejected."""
    # Cannot jump directly from DRAFT to ACTIVE
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_project_transition(
            ProjectStatus.DRAFT,
            ProjectStatus.ACTIVE,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        )

    # Cannot jump from SUBMITTED to COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_project_transition(
            ProjectStatus.SUBMITTED,
            ProjectStatus.COMPLETED,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        )

    # Cannot transition out of ARCHIVED
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_project_transition(
            ProjectStatus.ARCHIVED,
            ProjectStatus.ACTIVE,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        )


def test_project_unauthorized_actor_transition():
    """Verify unauthorized actors cannot trigger transitions."""
    # Regular donor cannot approve a project
    with pytest.raises(UnauthorizedTransitionError):
        StateMachineValidator.validate_project_transition(
            ProjectStatus.SUBMITTED,
            ProjectStatus.APPROVED,
            actor_platform_role=PlatformRole.DONOR
        )

    # Ordinary org member cannot submit project for review
    with pytest.raises(UnauthorizedTransitionError):
        StateMachineValidator.validate_project_transition(
            ProjectStatus.DRAFT,
            ProjectStatus.SUBMITTED,
            actor_org_role=OrganizationRole.MEMBER
        )
