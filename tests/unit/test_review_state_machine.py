"""
test_review_state_machine.py - Unit Tests for Phase 7 Review State Machines
===========================================================================
Validates state machines for:
1. ReviewCase transitions (SUBMITTED -> ASSIGNED -> IN_REVIEW -> etc.)
2. ReviewAssignment transitions (ASSIGNED -> ACCEPTED -> DECIDED, DECLINED, etc.)
3. Dispute transitions (OPEN -> UNDER_REVIEW -> RESOLVED, etc.)
4. Milestone review status updates and invalid transition enforcement.
"""

import pytest
from backend.app.core.constants import (
    ReviewCaseStatus,
    ReviewAssignmentStatus,
    DisputeStatus,
    MilestoneStatus,
    PlatformRole,
    OrganizationRole
)
from backend.app.core.state_machines import (
    StateMachineValidator,
    InvalidStateTransitionError,
    REVIEW_CASE_TRANSITIONS,
    REVIEW_ASSIGNMENT_TRANSITIONS,
    DISPUTE_TRANSITIONS,
    MILESTONE_TRANSITIONS
)


class TestReviewCaseStateMachine:
    """Validates ReviewCase status transitions."""

    def test_valid_case_transitions(self):
        """Verify normal progression transitions succeed."""
        # SUBMITTED -> ASSIGNED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.SUBMITTED, ReviewCaseStatus.ASSIGNED
        ) is True

        # ASSIGNED -> IN_REVIEW
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.ASSIGNED, ReviewCaseStatus.IN_REVIEW
        ) is True

        # IN_REVIEW -> AWAITING_SECOND_REVIEW
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.IN_REVIEW, ReviewCaseStatus.AWAITING_SECOND_REVIEW
        ) is True

        # AWAITING_SECOND_REVIEW -> APPROVED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.AWAITING_SECOND_REVIEW, ReviewCaseStatus.APPROVED
        ) is True

        # APPROVED -> CLOSED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.APPROVED, ReviewCaseStatus.CLOSED
        ) is True

    def test_escalation_and_revision_transitions(self):
        """Verify split decision escalation and revision flows."""
        # IN_REVIEW -> ESCALATED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.IN_REVIEW, ReviewCaseStatus.ESCALATED
        ) is True

        # ESCALATED -> APPROVED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.ESCALATED, ReviewCaseStatus.APPROVED
        ) is True

        # IN_REVIEW -> REVISION_REQUIRED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.IN_REVIEW, ReviewCaseStatus.REVISION_REQUIRED
        ) is True

        # REVISION_REQUIRED -> SUBMITTED (resubmission round)
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.REVISION_REQUIRED, ReviewCaseStatus.SUBMITTED
        ) is True

    def test_rejection_and_dispute_transitions(self):
        """Verify rejection and dispute workflow."""
        # IN_REVIEW -> REJECTED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.IN_REVIEW, ReviewCaseStatus.REJECTED
        ) is True

        # REJECTED -> DISPUTED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.REJECTED, ReviewCaseStatus.DISPUTED
        ) is True

        # DISPUTED -> RESOLVED
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.DISPUTED, ReviewCaseStatus.RESOLVED
        ) is True

        # RESOLVED -> SUBMITTED (re-review ordered)
        assert StateMachineValidator.validate_review_case_transition(
            ReviewCaseStatus.RESOLVED, ReviewCaseStatus.SUBMITTED
        ) is True

    def test_invalid_case_transitions_raise_error(self):
        """Verify invalid transitions raise InvalidStateTransitionError."""
        # Cannot jump from SUBMITTED directly to APPROVED
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_review_case_transition(
                ReviewCaseStatus.SUBMITTED, ReviewCaseStatus.APPROVED
            )

        # Cannot jump from CLOSED to SUBMITTED
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_review_case_transition(
                ReviewCaseStatus.CLOSED, ReviewCaseStatus.SUBMITTED
            )

        # Cannot jump from REJECTED directly to APPROVED
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_review_case_transition(
                ReviewCaseStatus.REJECTED, ReviewCaseStatus.APPROVED
            )


class TestReviewAssignmentStateMachine:
    """Validates ReviewTask assignment state transitions."""

    def test_assignment_lifecycle(self):
        """Verify normal assignment transitions."""
        # ASSIGNED -> ACCEPTED
        assert StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus.ASSIGNED, ReviewAssignmentStatus.ACCEPTED
        ) is True

        # ACCEPTED -> IN_PROGRESS
        assert StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus.ACCEPTED, ReviewAssignmentStatus.IN_PROGRESS
        ) is True

        # IN_PROGRESS -> DECIDED
        assert StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus.IN_PROGRESS, ReviewAssignmentStatus.DECIDED
        ) is True

    def test_assignment_decline_and_reassignment(self):
        """Verify reviewer decline and reassignment."""
        # ASSIGNED -> DECLINED
        assert StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus.ASSIGNED, ReviewAssignmentStatus.DECLINED
        ) is True

        # DECLINED -> REASSIGNED
        assert StateMachineValidator.validate_review_assignment_transition(
            ReviewAssignmentStatus.DECLINED, ReviewAssignmentStatus.REASSIGNED
        ) is True

    def test_invalid_assignment_transitions(self):
        """Verify DECIDED is terminal and invalid moves are blocked."""
        # DECIDED cannot move to ACCEPTED
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_review_assignment_transition(
                ReviewAssignmentStatus.DECIDED, ReviewAssignmentStatus.ACCEPTED
            )

        # REASSIGNED cannot move to DECIDED
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_review_assignment_transition(
                ReviewAssignmentStatus.REASSIGNED, ReviewAssignmentStatus.DECIDED
            )


class TestDisputeStateMachine:
    """Validates formal dispute lifecycle transitions."""

    def test_dispute_lifecycle(self):
        """Verify dispute progression."""
        # OPEN -> UNDER_REVIEW
        assert StateMachineValidator.validate_dispute_transition(
            DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW
        ) is True

        # UNDER_REVIEW -> AWAITING_INFORMATION
        assert StateMachineValidator.validate_dispute_transition(
            DisputeStatus.UNDER_REVIEW, DisputeStatus.AWAITING_INFORMATION
        ) is True

        # AWAITING_INFORMATION -> UNDER_REVIEW
        assert StateMachineValidator.validate_dispute_transition(
            DisputeStatus.AWAITING_INFORMATION, DisputeStatus.UNDER_REVIEW
        ) is True

        # UNDER_REVIEW -> RESOLVED
        assert StateMachineValidator.validate_dispute_transition(
            DisputeStatus.UNDER_REVIEW, DisputeStatus.RESOLVED
        ) is True

        # RESOLVED -> CLOSED
        assert StateMachineValidator.validate_dispute_transition(
            DisputeStatus.RESOLVED, DisputeStatus.CLOSED
        ) is True

    def test_invalid_dispute_transitions(self):
        """Verify invalid dispute jumps raise error."""
        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_dispute_transition(
                DisputeStatus.CLOSED, DisputeStatus.OPEN
            )

        with pytest.raises(InvalidStateTransitionError):
            StateMachineValidator.validate_dispute_transition(
                DisputeStatus.OPEN, DisputeStatus.RESOLVED
            )


class TestMilestoneReviewTransitions:
    """Validates Milestone transitions related to Phase 7 review states."""

    def test_milestone_review_states(self):
        """Verify milestone review state progression."""
        # ACTIVE -> EVIDENCE_SUBMITTED
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.ACTIVE,
            MilestoneStatus.EVIDENCE_SUBMITTED,
            actor_org_role=OrganizationRole.ADMIN
        ) is True

        # EVIDENCE_SUBMITTED -> VERIFICATION_PENDING
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.EVIDENCE_SUBMITTED,
            MilestoneStatus.VERIFICATION_PENDING,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        ) is True

        # VERIFICATION_PENDING -> COMPLETED (dual approval)
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.VERIFICATION_PENDING,
            MilestoneStatus.COMPLETED,
            actor_platform_role=PlatformRole.REVIEWER
        ) is True

        # VERIFICATION_PENDING -> REVISION_REQUIRED
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.VERIFICATION_PENDING,
            MilestoneStatus.REVISION_REQUIRED,
            actor_platform_role=PlatformRole.REVIEWER
        ) is True

        # REVISION_REQUIRED -> EVIDENCE_SUBMITTED (resubmission)
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.REVISION_REQUIRED,
            MilestoneStatus.EVIDENCE_SUBMITTED,
            actor_org_role=OrganizationRole.ADMIN
        ) is True

        # VERIFICATION_PENDING -> REJECTED
        assert StateMachineValidator.validate_milestone_transition(
            MilestoneStatus.VERIFICATION_PENDING,
            MilestoneStatus.REJECTED,
            actor_platform_role=PlatformRole.REVIEWER
        ) is True
