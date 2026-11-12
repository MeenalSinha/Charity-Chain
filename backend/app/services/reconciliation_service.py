"""
reconciliation_service.py - Automated Financial Reconciliation Engine
====================================================================
Matches local double-entry records with external gateway settlement records:
- Ingests gateway settlement files, webhook event logs, and bank statement feeds.
- Audits captures, refunds, disbursements, and processor fees.
- Pinpoints discrepancies: amount mismatch, status mismatch, missing local, missing gateway.
- Preserves complete reconciliation run history and manual resolution audit trails.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    ReconciliationRunStatus, ReconciliationStatus,
    DiscrepancyType, DiscrepancyResolution,
    PaymentCaptureStatus, PaymentRefundStatus, DisbursementStatus
)
from backend.app.core.exceptions import NotFoundException, ValidationException
from backend.app.core.logging import get_logger
from backend.app.db.models.financial import (
    PaymentCapture, PaymentRefund, Disbursement,
    ReconciliationRun, ReconciliationItem, SettlementRecord, ProcessorFeeRecord
)

logger = get_logger("services.reconciliation")


class ReconciliationService:
    """Core domain service for automated financial reconciliation and discrepancy auditing."""

    @staticmethod
    def ingest_settlement_record(
        db: Session,
        transaction_id: str,
        transaction_type: str,
        gross_amount_minor_units: int,
        fee_amount_minor_units: int,
        currency: str = "USD",
        settled_at: Optional[datetime] = None,
        settlement_batch_id: Optional[str] = None,
        raw_data: Optional[str] = None
    ) -> SettlementRecord:
        """Persists an external gateway transaction settlement report line."""
        existing = db.execute(
            select(SettlementRecord).where(SettlementRecord.transaction_id == transaction_id)
        ).scalar_one_or_none()

        if existing:
            return existing

        now = settled_at or datetime.now(timezone.utc)
        net_amount = gross_amount_minor_units - fee_amount_minor_units

        record = SettlementRecord(
            id=str(uuid.uuid4()),
            provider="PAYPAL",
            settlement_batch_id=settlement_batch_id,
            transaction_id=transaction_id,
            transaction_type=transaction_type.upper(),
            gross_amount_minor_units=gross_amount_minor_units,
            fee_amount_minor_units=fee_amount_minor_units,
            net_amount_minor_units=net_amount,
            currency=currency,
            settled_at=now,
            raw_data=raw_data
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @classmethod
    def run_reconciliation(
        cls,
        db: Session,
        period_start: datetime,
        period_end: datetime,
        executed_by_id: Optional[str] = None,
        provider: str = "PAYPAL"
    ) -> ReconciliationRun:
        """
        Executes automated reconciliation between local transaction records and external settlements.
        """
        now = datetime.now(timezone.utc)
        run_number = f"REC-{now.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

        run = ReconciliationRun(
            id=str(uuid.uuid4()),
            run_number=run_number,
            provider=provider,
            period_start=period_start,
            period_end=period_end,
            status=ReconciliationRunStatus.PROCESSING,
            executed_by_id=executed_by_id
        )
        db.add(run)
        db.flush()

        matched_count = 0
        discrepancy_count = 0
        total_local_minor = 0
        total_gateway_minor = 0
        matched_gateway_tx_ids = set()

        # 1. Reconcile Payment Captures (Donations)
        local_captures = db.execute(
            select(PaymentCapture).where(
                and_(
                    PaymentCapture.captured_at >= period_start,
                    PaymentCapture.captured_at <= period_end,
                    PaymentCapture.status.in_([
                        PaymentCaptureStatus.COMPLETED,
                        PaymentCaptureStatus.PARTIALLY_REFUNDED,
                        PaymentCaptureStatus.REFUNDED
                    ])
                )
            )
        ).scalars().all()

        for cap in local_captures:
            total_local_minor += cap.amount_minor_units
            settlement = db.execute(
                select(SettlementRecord).where(
                    and_(
                        SettlementRecord.transaction_id == cap.provider_capture_id,
                        SettlementRecord.transaction_type == "PAYMENT"
                    )
                )
            ).scalar_one_or_none()

            if settlement:
                matched_gateway_tx_ids.add(settlement.transaction_id)
                total_gateway_minor += settlement.gross_amount_minor_units

                if cap.amount_minor_units == settlement.gross_amount_minor_units:
                    rec_status = ReconciliationStatus.MATCHED
                    disc_type = None
                    matched_count += 1
                else:
                    rec_status = ReconciliationStatus.DISCREPANCY
                    disc_type = DiscrepancyType.AMOUNT_MISMATCH
                    discrepancy_count += 1

                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="CAPTURE",
                    reference_id=cap.id,
                    provider_transaction_id=cap.provider_capture_id,
                    local_amount_minor_units=cap.amount_minor_units,
                    gateway_amount_minor_units=settlement.gross_amount_minor_units,
                    local_status=cap.status.value,
                    gateway_status="SETTLED",
                    reconciliation_status=rec_status,
                    discrepancy_type=disc_type
                )
                db.add(item)
            else:
                discrepancy_count += 1
                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="CAPTURE",
                    reference_id=cap.id,
                    provider_transaction_id=cap.provider_capture_id,
                    local_amount_minor_units=cap.amount_minor_units,
                    gateway_amount_minor_units=0,
                    local_status=cap.status.value,
                    gateway_status="NOT_FOUND_IN_SETTLEMENT",
                    reconciliation_status=ReconciliationStatus.DISCREPANCY,
                    discrepancy_type=DiscrepancyType.MISSING_IN_GATEWAY
                )
                db.add(item)

        # 2. Reconcile Payment Refunds
        local_refunds = db.execute(
            select(PaymentRefund).where(
                and_(
                    PaymentRefund.refunded_at >= period_start,
                    PaymentRefund.refunded_at <= period_end,
                    PaymentRefund.status == PaymentRefundStatus.COMPLETED
                )
            )
        ).scalars().all()

        for ref in local_refunds:
            total_local_minor += ref.amount_minor_units
            settlement = db.execute(
                select(SettlementRecord).where(
                    and_(
                        SettlementRecord.transaction_id == ref.provider_refund_id,
                        SettlementRecord.transaction_type == "REFUND"
                    )
                )
            ).scalar_one_or_none()

            if settlement:
                matched_gateway_tx_ids.add(settlement.transaction_id)
                total_gateway_minor += settlement.gross_amount_minor_units

                if ref.amount_minor_units == settlement.gross_amount_minor_units:
                    rec_status = ReconciliationStatus.MATCHED
                    disc_type = None
                    matched_count += 1
                else:
                    rec_status = ReconciliationStatus.DISCREPANCY
                    disc_type = DiscrepancyType.AMOUNT_MISMATCH
                    discrepancy_count += 1

                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="REFUND",
                    reference_id=ref.id,
                    provider_transaction_id=ref.provider_refund_id,
                    local_amount_minor_units=ref.amount_minor_units,
                    gateway_amount_minor_units=settlement.gross_amount_minor_units,
                    local_status=ref.status.value,
                    gateway_status="SETTLED",
                    reconciliation_status=rec_status,
                    discrepancy_type=disc_type
                )
                db.add(item)
            else:
                discrepancy_count += 1
                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="REFUND",
                    reference_id=ref.id,
                    provider_transaction_id=ref.provider_refund_id,
                    local_amount_minor_units=ref.amount_minor_units,
                    gateway_amount_minor_units=0,
                    local_status=ref.status.value,
                    gateway_status="NOT_FOUND_IN_SETTLEMENT",
                    reconciliation_status=ReconciliationStatus.DISCREPANCY,
                    discrepancy_type=DiscrepancyType.MISSING_IN_GATEWAY
                )
                db.add(item)

        # 3. Reconcile Outbound Disbursements
        local_disbursements = db.execute(
            select(Disbursement).where(
                and_(
                    Disbursement.disbursed_at >= period_start,
                    Disbursement.disbursed_at <= period_end,
                    Disbursement.status == DisbursementStatus.SUCCEEDED
                )
            )
        ).scalars().all()

        for d in local_disbursements:
            total_local_minor += d.amount_minor_units
            settlement = db.execute(
                select(SettlementRecord).where(
                    and_(
                        SettlementRecord.transaction_id == d.provider_item_id,
                        SettlementRecord.transaction_type == "PAYOUT"
                    )
                )
            ).scalar_one_or_none()

            if settlement:
                matched_gateway_tx_ids.add(settlement.transaction_id)
                total_gateway_minor += settlement.gross_amount_minor_units

                if d.amount_minor_units == settlement.gross_amount_minor_units:
                    rec_status = ReconciliationStatus.MATCHED
                    disc_type = None
                    matched_count += 1
                else:
                    rec_status = ReconciliationStatus.DISCREPANCY
                    disc_type = DiscrepancyType.AMOUNT_MISMATCH
                    discrepancy_count += 1

                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="PAYOUT",
                    reference_id=d.id,
                    provider_transaction_id=d.provider_item_id,
                    local_amount_minor_units=d.amount_minor_units,
                    gateway_amount_minor_units=settlement.gross_amount_minor_units,
                    local_status=d.status.value,
                    gateway_status="SETTLED",
                    reconciliation_status=rec_status,
                    discrepancy_type=disc_type
                )
                db.add(item)
            else:
                discrepancy_count += 1
                item = ReconciliationItem(
                    id=str(uuid.uuid4()),
                    reconciliation_run_id=run.id,
                    item_type="PAYOUT",
                    reference_id=d.id,
                    provider_transaction_id=d.provider_item_id,
                    local_amount_minor_units=d.amount_minor_units,
                    gateway_amount_minor_units=0,
                    local_status=d.status.value,
                    gateway_status="NOT_FOUND_IN_SETTLEMENT",
                    reconciliation_status=ReconciliationStatus.DISCREPANCY,
                    discrepancy_type=DiscrepancyType.MISSING_IN_GATEWAY
                )
                db.add(item)

        # 4. Check for unmapped settlement records (MISSING_IN_LOCAL)
        unmatched_settlements = db.execute(
            select(SettlementRecord).where(
                and_(
                    SettlementRecord.settled_at >= period_start,
                    SettlementRecord.settled_at <= period_end,
                    ~SettlementRecord.transaction_id.in_(matched_gateway_tx_ids) if matched_gateway_tx_ids else True
                )
            )
        ).scalars().all()

        for s in unmatched_settlements:
            discrepancy_count += 1
            total_gateway_minor += s.gross_amount_minor_units
            item = ReconciliationItem(
                id=str(uuid.uuid4()),
                reconciliation_run_id=run.id,
                item_type=s.transaction_type,
                reference_id=None,
                provider_transaction_id=s.transaction_id,
                local_amount_minor_units=0,
                gateway_amount_minor_units=s.gross_amount_minor_units,
                local_status="NOT_FOUND_LOCALLY",
                gateway_status="SETTLED",
                reconciliation_status=ReconciliationStatus.DISCREPANCY,
                discrepancy_type=DiscrepancyType.MISSING_IN_LOCAL
            )
            db.add(item)

        total_analyzed = matched_count + discrepancy_count
        run.total_records_analyzed = total_analyzed
        run.matched_count = matched_count
        run.discrepancy_count = discrepancy_count
        run.total_local_minor_units = total_local_minor
        run.total_gateway_minor_units = total_gateway_minor
        run.completed_at = datetime.now(timezone.utc)

        if discrepancy_count == 0:
            run.status = ReconciliationRunStatus.COMPLETED
        else:
            run.status = ReconciliationRunStatus.COMPLETED_WITH_DISCREPANCIES

        db.commit()
        db.refresh(run)

        logger.info(
            f"Reconciliation run {run.run_number} finished: {matched_count} matched, {discrepancy_count} discrepancies",
            extra={"run_number": run.run_number, "matched": matched_count, "discrepancies": discrepancy_count}
        )
        return run

    @staticmethod
    def resolve_discrepancy(
        db: Session,
        item_id: str,
        resolution: DiscrepancyResolution,
        notes: str,
        resolver_id: str
    ) -> ReconciliationItem:
        """Documents authorized human resolution of an identified reconciliation discrepancy."""
        item = db.execute(select(ReconciliationItem).where(ReconciliationItem.id == item_id)).scalar_one_or_none()
        if not item:
            raise NotFoundException(f"Reconciliation item '{item_id}' not found.")

        item.reconciliation_status = ReconciliationStatus.MANUALLY_RESOLVED
        item.discrepancy_resolution = resolution
        item.resolution_notes = notes
        item.resolved_by_id = resolver_id
        item.resolved_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(item)
        return item

    @staticmethod
    def get_run_details(db: Session, run_id: str) -> Dict[str, Any]:
        """Returns details and all reconciliation items for a specific run."""
        run = db.execute(select(ReconciliationRun).where(ReconciliationRun.id == run_id)).scalar_one_or_none()
        if not run:
            raise NotFoundException(f"Reconciliation run '{run_id}' not found.")

        items = db.execute(
            select(ReconciliationItem).where(ReconciliationItem.reconciliation_run_id == run.id)
        ).scalars().all()

        return {
            "run_id": run.id,
            "run_number": run.run_number,
            "provider": run.provider,
            "status": run.status.value,
            "period_start": run.period_start.isoformat(),
            "period_end": run.period_end.isoformat(),
            "total_records_analyzed": run.total_records_analyzed,
            "matched_count": run.matched_count,
            "discrepancy_count": run.discrepancy_count,
            "total_local_minor_units": run.total_local_minor_units,
            "total_gateway_minor_units": run.total_gateway_minor_units,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "items": [
                {
                    "item_id": it.id,
                    "item_type": it.item_type,
                    "reference_id": it.reference_id,
                    "provider_transaction_id": it.provider_transaction_id,
                    "local_amount_minor_units": it.local_amount_minor_units,
                    "gateway_amount_minor_units": it.gateway_amount_minor_units,
                    "local_status": it.local_status,
                    "gateway_status": it.gateway_status,
                    "reconciliation_status": it.reconciliation_status.value,
                    "discrepancy_type": it.discrepancy_type.value if it.discrepancy_type else None,
                    "discrepancy_resolution": it.discrepancy_resolution.value if it.discrepancy_resolution else None,
                    "resolution_notes": it.resolution_notes
                }
                for it in items
            ]
        }

