"""
test_db_models.py - Tests for SQLAlchemy Domain Model Definitions
"""

from backend.app.db.models import (
    Base, User, Organization, NGOProfile,
    Project, Milestone, ProjectLocation,
    Donation, PaymentOrder, PaymentAuthorization,
    EvidenceFile, VerificationReport, ImpactCertificate
)


def test_models_metadata_tables_present():
    tables = Base.metadata.tables
    expected_tables = [
        "users", "user_profiles", "organizations", "organization_memberships",
        "ngo_profiles", "project_categories", "projects", "milestones",
        "project_locations", "donations", "payment_orders",
        "payment_authorizations", "payment_captures", "payment_refunds",
        "disbursements", "evidence_files", "verification_reports",
        "verification_check_records", "review_tasks", "review_decision_records",
        "impact_certificates", "audit_events"
    ]
    for table_name in expected_tables:
        assert table_name in tables, f"Expected table '{table_name}' in metadata"


def test_model_primary_keys_and_mixins():
    user = User(email="test@charity.org", hashed_password="xxx", full_name="Test User")
    assert user.id is not None
    assert user.created_at is not None
    assert user.is_active is True

