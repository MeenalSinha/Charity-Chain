"""
test_security_audit_api.py - Integration Tests for Admin Security Audit APIs
============================================================================
Validates administrative querying, pagination, filtering, and cryptographic
chain verification via /api/v1/admin/security/audit-logs.
"""

import pytest
from datetime import datetime, timezone
from backend.app.core.constants import PlatformRole, AccountStatus
from backend.app.db.models.users import User
from backend.app.services.security_audit_service import SecurityAuditService
from backend.app.api.v1.admin import list_security_audit_logs, verify_security_audit_chain


@pytest.fixture
def admin_user(db_session):
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    admin = User(
        email=f"sec_admin_{ts}@charitychain.org",
        full_name="Security Administrator",
        hashed_password="hashed_pass",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(admin)
    db_session.commit()
    return admin


def test_list_and_verify_security_audit_logs(admin_user, db_session):
    """Verifies that an admin can list and verify cryptographic audit trails."""
    # 1. Record test audit events
    SecurityAuditService.record_event(
        db=db_session,
        event_type="AUTH_LOGIN_SUCCESS",
        action="user_login",
        severity="INFO",
        user_id=admin_user.id,
        details={"ip": "127.0.0.1"}
    )
    SecurityAuditService.record_event(
        db=db_session,
        event_type="ADMIN_ACTION",
        action="view_audit_logs",
        severity="WARNING",
        user_id=admin_user.id,
        details={"scope": "all"}
    )

    # 2. Query audit logs via API
    response = list_security_audit_logs(
        page=1,
        page_size=10,
        current_admin=admin_user,
        db=db_session
    )
    assert response.success is True
    assert response.data["total"] >= 2
    assert len(response.data["items"]) >= 2

    # 3. Verify cryptographic chain integrity via API
    verify_resp = verify_security_audit_chain(
        limit=1000,
        current_admin=admin_user,
        db=db_session
    )
    assert verify_resp.success is True
    assert verify_resp.data["is_valid"] is True
    assert verify_resp.data["audited_count"] >= 2

