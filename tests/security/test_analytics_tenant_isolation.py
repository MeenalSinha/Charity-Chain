"""
test_analytics_tenant_isolation.py - Security & IDOR Isolation Tests
====================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization, OrganizationMembership
from backend.app.core.constants import PlatformRole, OrganizationRole, AccountStatus
from backend.app.core.exceptions import ForbiddenException
from backend.app.api.v1.analytics import verify_ngo_access


def test_cross_organization_analytics_access_blocked(db_session):
    """Verifies that an NGO admin cannot access another organization's private analytics (IDOR / Tenant Isolation)."""
    # NGO A Admin
    user_a = User(
        id=str(uuid.uuid4()),
        email="admin_a@ngo-alpha.org",
        full_name="Admin A",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org_a = Organization(id=str(uuid.uuid4()), name="NGO Alpha", creator_id=user_a.id)
    mem_a = OrganizationMembership(
        id=str(uuid.uuid4()), user_id=user_a.id, organization_id=org_a.id, role=OrganizationRole.OWNER, is_active=True
    )

    # NGO B
    user_b = User(
        id=str(uuid.uuid4()),
        email="admin_b@ngo-beta.org",
        full_name="Admin B",
        hashed_password="mock_hashed_password",
        role=PlatformRole.DONOR,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    org_b = Organization(id=str(uuid.uuid4()), name="NGO Beta", creator_id=user_b.id)
    mem_b = OrganizationMembership(
        id=str(uuid.uuid4()), user_id=user_b.id, organization_id=org_b.id, role=OrganizationRole.OWNER, is_active=True
    )

    db_session.add_all([user_a, org_a, mem_a, user_b, org_b, mem_b])
    db_session.commit()

    # User A attempting to access Org A -> SUCCESS
    verify_ngo_access(organization_id=org_a.id, current_user=user_a, db=db_session)

    # User A attempting to access Org B -> FORBIDDEN
    with pytest.raises(ForbiddenException):
        verify_ngo_access(organization_id=org_b.id, current_user=user_a, db=db_session)
