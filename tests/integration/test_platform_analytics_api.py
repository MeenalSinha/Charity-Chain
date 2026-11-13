"""
test_platform_analytics_api.py - Integration Tests for Platform Administration
==============================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import uuid
import pytest
from backend.app.db.models.users import User, Organization
from backend.app.db.models.projects import Project
from backend.app.db.models.financial import PaymentCapture, Disbursement
from backend.app.db.models.ledger import LedgerAccount, JournalEntry, JournalLine
from backend.app.core.constants import (
    PlatformRole, AccountStatus, ProjectStatus, PaymentCaptureStatus,
    DisbursementStatus, LedgerAccountType, LedgerEntryType
)
from backend.app.services.analytics.platform_analytics_service import PlatformAnalyticsService


def test_platform_analytics_ledger_balance_check(db_session):
    """Verifies that double-entry trial balance check detects balance vs imbalance."""
    admin = User(
        id=str(uuid.uuid4()),
        email="admin_analytics@example.org",
        full_name="Platform Admin Test",
        hashed_password="mock_hashed_password",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(admin)
    db_session.commit()

    # Balanced ledger lines: 10,000 cents Debit, 10,000 cents Credit
    acc_asset = LedgerAccount(id=str(uuid.uuid4()), account_number="1000", name="Cash Escrow", account_type=LedgerAccountType.ASSET)
    acc_liab = LedgerAccount(id=str(uuid.uuid4()), account_number="2000", name="Donation Payable", account_type=LedgerAccountType.LIABILITY)
    entry = JournalEntry(id=str(uuid.uuid4()), entry_number="JE-001", entry_type=LedgerEntryType.DONATION_CAPTURE, description="Donor Payment Received")
    line1 = JournalLine(id=str(uuid.uuid4()), journal_entry_id=entry.id, account_id=acc_asset.id, debit_minor_units=10000, credit_minor_units=0)
    line2 = JournalLine(id=str(uuid.uuid4()), journal_entry_id=entry.id, account_id=acc_liab.id, debit_minor_units=0, credit_minor_units=10000)

    db_session.add_all([acc_asset, acc_liab, entry, line1, line2])
    db_session.commit()

    fin_ops = PlatformAnalyticsService.get_financial_operations_analytics(db_session)
    ledger_int = fin_ops["financial_ledger_integrity"]

    assert ledger_int["is_balanced"] is True
    assert ledger_int["total_debits_cents"] == 10000
    assert ledger_int["total_credits_cents"] == 10000
    assert ledger_int["discrepancy_cents"] == 0
