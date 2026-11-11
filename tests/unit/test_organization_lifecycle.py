"""
test_organization_lifecycle.py - Unit tests for Organization State Machine and Lifecycle
=======================================================================================
Verifies valid and invalid organization status transitions, actor permissions,
revision workflows, and data validations.
"""

import pytest
from backend.app.core.constants import OrganizationStatus, PlatformRole, OrganizationRole
from backend.app.core.state_machines import StateMachineValidator
from backend.app.core.exceptions import InvalidStateTransitionError, UnauthorizedTransitionError
from backend.app.api.schemas import CreateOrganizationRequest, UpdateOrganizationRequest, OrganizationType


def test_organization_status_valid_transitions():
    """Verify standard legitimate organization lifecycle transitions."""
    # DRAFT -> PENDING (Actor: Organization Admin/Owner)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.DRAFT,
        OrganizationStatus.PENDING,
        actor_org_role=OrganizationRole.OWNER
    )

    # PENDING -> VERIFIED (Actor: Reviewer)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.PENDING,
        OrganizationStatus.VERIFIED,
        actor_platform_role=PlatformRole.REVIEWER
    )

    # PENDING -> REVISION_REQUIRED (Actor: Reviewer)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.PENDING,
        OrganizationStatus.REVISION_REQUIRED,
        actor_platform_role=PlatformRole.REVIEWER
    )

    # REVISION_REQUIRED -> PENDING (Actor: Organization Owner after uploading revisions)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.REVISION_REQUIRED,
        OrganizationStatus.PENDING,
        actor_org_role=OrganizationRole.OWNER
    )

    # PENDING -> REJECTED (Actor: Reviewer)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.PENDING,
        OrganizationStatus.REJECTED,
        actor_platform_role=PlatformRole.REVIEWER
    )

    # VERIFIED -> SUSPENDED (Actor: Platform Admin)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.VERIFIED,
        OrganizationStatus.SUSPENDED,
        actor_platform_role=PlatformRole.PLATFORM_ADMIN
    )

    # SUSPENDED -> VERIFIED (Actor: Platform Admin reactivates)
    assert StateMachineValidator.validate_organization_transition(
        OrganizationStatus.SUSPENDED,
        OrganizationStatus.VERIFIED,
        actor_platform_role=PlatformRole.PLATFORM_ADMIN
    )


def test_organization_status_invalid_transitions():
    """Verify disallowed state transitions raise InvalidStateTransitionError."""
    # Cannot jump from DRAFT directly to VERIFIED
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_organization_transition(
            OrganizationStatus.DRAFT,
            OrganizationStatus.VERIFIED,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        )

    # Cannot transition from SUSPENDED directly to DRAFT
    with pytest.raises(InvalidStateTransitionError):
        StateMachineValidator.validate_organization_transition(
            OrganizationStatus.SUSPENDED,
            OrganizationStatus.DRAFT,
            actor_platform_role=PlatformRole.PLATFORM_ADMIN
        )


def test_organization_transition_unauthorized_actor():
    """Verify transitions with unauthorized roles are blocked."""
    # A regular donor cannot verify an organization
    with pytest.raises(UnauthorizedTransitionError):
        StateMachineValidator.validate_organization_transition(
            OrganizationStatus.PENDING,
            OrganizationStatus.VERIFIED,
            actor_platform_role=PlatformRole.DONOR
        )

    # An ordinary org member cannot submit the organization for verification
    with pytest.raises(UnauthorizedTransitionError):
        StateMachineValidator.validate_organization_transition(
            OrganizationStatus.DRAFT,
            OrganizationStatus.PENDING,
            actor_org_role=OrganizationRole.MEMBER
        )


def test_create_organization_schema_validation():
    """Verify CreateOrganizationRequest validates ISO country codes and required fields."""
    req = CreateOrganizationRequest(
        legal_name="Global Care Foundation",
        organization_type=OrganizationType.NON_PROFIT,
        registration_authority="IRS",
        registration_number="REG-12345",
        country_of_registration="US",
        registered_address="100 Main St, New York, NY",
        operating_address="100 Main St, New York, NY",
        primary_operating_region="Global",
        mission_statement="Empowering vulnerable communities through verified aid."
    )
    assert req.legal_name == "Global Care Foundation"
    assert req.country_of_registration == "US"
    assert req.organization_type == OrganizationType.NON_PROFIT
