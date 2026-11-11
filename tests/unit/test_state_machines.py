"""
test_state_machines.py - Domain State Transition & Permission Tests
"""

import pytest
from backend.app.core.constants import MilestoneStatus, ProjectStatus, UserRole
from backend.app.core.state_machines import StateMachineValidator
from backend.app.core.exceptions import InvalidStateTransitionError, UnauthorizedTransitionError


def test_milestone_valid_transition():
    # NGO submits drafted milestone
    StateMachineValidator.validate_milestone_transition(
        current=MilestoneStatus.DRAFT,
        target=MilestoneStatus.SUBMITTED,
        actor_role=UserRole.NGO_ADMIN
    )


def test_milestone_invalid_transition():
    # Cannot jump from DRAFT directly to COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_milestone_transition(
            current=MilestoneStatus.DRAFT,
            target=MilestoneStatus.COMPLETED,
            actor_role=UserRole.REVIEWER
        )


def test_milestone_unauthorized_actor():
    # Donor cannot approve a milestone
    with pytest.raises(UnauthorizedTransitionError):
        StateMachineValidator.validate_milestone_transition(
            current=MilestoneStatus.UNDER_REVIEW,
            target=MilestoneStatus.APPROVED,
            actor_role=UserRole.DONOR
        )
    
    # Reviewer CAN approve a milestone
    StateMachineValidator.validate_milestone_transition(
        current=MilestoneStatus.UNDER_REVIEW,
        target=MilestoneStatus.APPROVED,
        actor_role=UserRole.REVIEWER
    )


def test_project_transitions():
    # Reviewer approves submitted project
    StateMachineValidator.validate_project_transition(
        current=ProjectStatus.SUBMITTED,
        target=ProjectStatus.APPROVED,
        actor_role=UserRole.REVIEWER
    )
    
    # Disallowed transition
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_project_transition(
            current=ProjectStatus.COMPLETED,
            target=ProjectStatus.ACTIVE,
            actor_role=UserRole.PLATFORM_ADMIN
        )

