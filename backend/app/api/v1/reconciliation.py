"""
reconciliation.py - Financial Reconciliation & Discrepancy Auditing API Endpoints
================================================================================
Endpoints to execute automated reconciliation runs, query discrepancy findings,
ingest external settlement lines, and document discrepancy resolutions.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    StandardResponse, ReconciliationRunPayload, DiscrepancyResolvePayload
)
from backend.app.core.constants import PlatformRole
from backend.app.core.exceptions import ForbiddenException
from backend.app.core.permissions import get_current_user
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.services.reconciliation_service import ReconciliationService

router = APIRouter(prefix="/reconciliation", tags=["Financial Reconciliation"])


@router.post("/runs", response_model=StandardResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def trigger_reconciliation_run(
    payload: ReconciliationRunPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initiates an automated reconciliation session across captures, refunds, payouts, and settlements.
    Restricted to Platform Admin, Finance Operator, and Finance Auditor.
    """
    allowed_roles = (
        PlatformRole.PLATFORM_ADMIN,
        PlatformRole.FINANCE_OPERATOR,
        PlatformRole.FINANCE_AUDITOR
    )
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Access denied: Financial auditor or finance role required.")

    run = ReconciliationService.run_reconciliation(
        db=db,
        period_start=payload.period_start,
        period_end=payload.period_end,
        executed_by_id=current_user.id,
        provider=payload.provider or "PAYPAL"
    )

    details = ReconciliationService.get_run_details(db, run.id)
    return StandardResponse(
        data=details,
        message=f"Reconciliation run {run.run_number} completed with status: {run.status.value}"
    )


@router.get("/runs/{run_id}", response_model=StandardResponse[Dict[str, Any]])
def get_reconciliation_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full details and itemized audit list for a reconciliation run."""
    allowed_roles = (
        PlatformRole.PLATFORM_ADMIN,
        PlatformRole.FINANCE_OPERATOR,
        PlatformRole.FINANCE_APPROVER,
        PlatformRole.FINANCE_AUDITOR
    )
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Access denied: Financial auditor or finance role required.")

    details = ReconciliationService.get_run_details(db, run_id)
    return StandardResponse(data=details)


@router.post("/items/{item_id}/resolve", response_model=StandardResponse[Dict[str, Any]])
def resolve_discrepancy_item(
    item_id: str,
    payload: DiscrepancyResolvePayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Documents authorized resolution of an audited financial discrepancy.
    Restricted to Platform Admin and Finance Operator.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance operators can resolve discrepancies.")

    resolved_item = ReconciliationService.resolve_discrepancy(
        db=db,
        item_id=item_id,
        resolution=payload.resolution,
        notes=payload.notes,
        resolver_id=current_user.id
    )

    return StandardResponse(
        data={
            "item_id": resolved_item.id,
            "reconciliation_status": resolved_item.reconciliation_status.value,
            "resolution": resolved_item.discrepancy_resolution.value if resolved_item.discrepancy_resolution else None,
            "resolution_notes": resolved_item.resolution_notes,
            "resolved_at": resolved_item.resolved_at.isoformat() if resolved_item.resolved_at else None
        },
        message="Discrepancy resolved successfully"
    )


@router.post("/settlements/ingest", response_model=StandardResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def ingest_settlement_record(
    transaction_id: str,
    transaction_type: str,
    gross_amount_minor_units: int,
    fee_amount_minor_units: int = 0,
    currency: str = "USD",
    settled_at: Optional[datetime] = None,
    settlement_batch_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ingests an external settlement report line from PayPal or bank file."""
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Access denied.")

    record = ReconciliationService.ingest_settlement_record(
        db=db,
        transaction_id=transaction_id,
        transaction_type=transaction_type,
        gross_amount_minor_units=gross_amount_minor_units,
        fee_amount_minor_units=fee_amount_minor_units,
        currency=currency,
        settled_at=settled_at,
        settlement_batch_id=settlement_batch_id
    )

    return StandardResponse(
        data={
            "id": record.id,
            "transaction_id": record.transaction_id,
            "transaction_type": record.transaction_type,
            "gross_amount_minor_units": record.gross_amount_minor_units,
            "net_amount_minor_units": record.net_amount_minor_units
        },
        message="Settlement record ingested"
    )

