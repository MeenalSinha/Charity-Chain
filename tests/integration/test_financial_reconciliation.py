"""
test_financial_reconciliation.py - Automated Financial Reconciliation Integration Tests
======================================================================================
Tests:
1. Ingesting external gateway settlement records.
2. Automated reconciliation run across captures, refunds, and payouts.
3. Identifying matching transactions and discrepancies (amount mismatch, missing records).
4. Documenting authorized human resolution for discrepancies.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from backend.app.core.constants import (
    PaymentCaptureStatus, ReconciliationStatus, DiscrepancyType,
    DiscrepancyResolution, ReconciliationRunStatus
)
from backend.app.db.models.financial import (
    PaymentCapture, SettlementRecord, ReconciliationRun, ReconciliationItem
)
from backend.app.services.reconciliation_service import ReconciliationService


def test_automated_reconciliation_and_discrepancy_detection(db_session):
    """
    Sets up local captures and settlement records:
    - Cap 1 ($100.00) matches Settlement 1 ($100.00) -> MATCHED
    - Cap 2 ($200.00) has Settlement 2 with amount $190.00 ($10 fee/diff) -> DISCREPANCY (AMOUNT_MISMATCH)
    - Cap 3 ($50.00) has no settlement record -> DISCREPANCY (MISSING_IN_GATEWAY)
    - Settlement 4 ($80.00) has no local capture -> DISCREPANCY (MISSING_IN_LOCAL)
    """
    now = datetime.now(timezone.utc)
    t_start = now - timedelta(days=2)
    t_end = now + timedelta(days=2)

    # 1. Local Captures
    c1 = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-REC-01", amount_minor_units=10000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)
    c2 = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-REC-02", amount_minor_units=20000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)
    c3 = PaymentCapture(id=str(uuid.uuid4()), provider_capture_id="CAP-REC-03", amount_minor_units=5000, currency="USD", status=PaymentCaptureStatus.COMPLETED, captured_at=now)
    db_session.add_all([c1, c2, c3])

    # 2. External Settlements
    s1 = SettlementRecord(id=str(uuid.uuid4()), transaction_id="CAP-REC-01", transaction_type="PAYMENT", gross_amount_minor_units=10000, fee_amount_minor_units=300, net_amount_minor_units=9700, currency="USD", settled_at=now)
    s2 = SettlementRecord(id=str(uuid.uuid4()), transaction_id="CAP-REC-02", transaction_type="PAYMENT", gross_amount_minor_units=19000, fee_amount_minor_units=500, net_amount_minor_units=18500, currency="USD", settled_at=now)  # Amount mismatch: 19000 vs 20000
    s4 = SettlementRecord(id=str(uuid.uuid4()), transaction_id="CAP-UNKNOWN-99", transaction_type="PAYMENT", gross_amount_minor_units=8000, fee_amount_minor_units=200, net_amount_minor_units=7800, currency="USD", settled_at=now)
    db_session.add_all([s1, s2, s4])
    db_session.commit()

    # 3. Run reconciliation
    run = ReconciliationService.run_reconciliation(
        db=db_session,
        period_start=t_start,
        period_end=t_end,
        executed_by_id="auditor-1"
    )

    assert run.status == ReconciliationRunStatus.COMPLETED_WITH_DISCREPANCIES
    assert run.matched_count == 1
    assert run.discrepancy_count == 3
    assert run.total_records_analyzed == 4

    details = ReconciliationService.get_run_details(db_session, run.id)
    items = details["items"]

    # Verify matching item
    m_item = next(it for it in items if it["provider_transaction_id"] == "CAP-REC-01")
    assert m_item["reconciliation_status"] == ReconciliationStatus.MATCHED.value

    # Verify amount mismatch item
    diff_item = next(it for it in items if it["provider_transaction_id"] == "CAP-REC-02")
    assert diff_item["reconciliation_status"] == ReconciliationStatus.DISCREPANCY.value
    assert diff_item["discrepancy_type"] == DiscrepancyType.AMOUNT_MISMATCH.value

    # 4. Resolve discrepancy on diff_item
    resolved = ReconciliationService.resolve_discrepancy(
        db=db_session,
        item_id=diff_item["item_id"],
        resolution=DiscrepancyResolution.MANUALLY_RECONCILED,
        notes="Gateway deducted $10 processing adjustment confirmed with statement",
        resolver_id="auditor-1"
    )

    assert resolved.reconciliation_status == ReconciliationStatus.MANUALLY_RESOLVED
    assert resolved.discrepancy_resolution == DiscrepancyResolution.MANUALLY_RECONCILED
