"""
ledger.py - Double-Entry Accounting Ledger & Trial Balance API Endpoints
========================================================================
Exposes endpoints for auditing double-entry accounts, running trial balances,
inspecting account transactional statements, and recording manual journal adjustments.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.api.schemas import (
    StandardResponse, LedgerEntryPayload
)
from backend.app.core.constants import PlatformRole
from backend.app.core.exceptions import ForbiddenException
from backend.app.core.permissions import get_current_user, require_platform_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.services.ledger_service import LedgerService

router = APIRouter(prefix="/ledger", tags=["Double-Entry Ledger"])


@router.get("/trial-balance", response_model=StandardResponse[Dict[str, Any]])
def get_trial_balance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Computes complete double-entry Trial Balance across all accounts.
    Accessible to Platform Admin, Finance Operator, Finance Approver, and Finance Auditor.
    """
    allowed_roles = (
        PlatformRole.PLATFORM_ADMIN,
        PlatformRole.FINANCE_OPERATOR,
        PlatformRole.FINANCE_APPROVER,
        PlatformRole.FINANCE_AUDITOR
    )
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Access denied: Financial auditor or finance role required.")

    tb = LedgerService.get_trial_balance(db)
    return StandardResponse(data=tb, message="Trial balance generated")


@router.get("/accounts/{account_number}/statement", response_model=StandardResponse[Dict[str, Any]])
def get_account_statement(
    account_number: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves full transaction history and running balance for a specific ledger account.
    """
    allowed_roles = (
        PlatformRole.PLATFORM_ADMIN,
        PlatformRole.FINANCE_OPERATOR,
        PlatformRole.FINANCE_APPROVER,
        PlatformRole.FINANCE_AUDITOR
    )
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Access denied: Financial role required.")

    stmt = LedgerService.get_account_statement(db, account_number)
    return StandardResponse(data=stmt)


@router.post("/entries", response_model=StandardResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def post_manual_journal_entry(
    payload: LedgerEntryPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Posts a balanced double-entry journal entry.
    Enforces sum(debits) == sum(credits).
    Restricted to Platform Admin and Finance Operator.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance operators can post manual journal entries.")

    lines_dict = [l.model_dump() for l in payload.lines]
    entry = LedgerService.post_journal_entry(
        db=db,
        entry_type=payload.entry_type,
        description=payload.description,
        lines=lines_dict,
        reference_id=payload.reference_id,
        reference_type=payload.reference_type,
        actor_id=current_user.id
    )

    return StandardResponse(
        data={
            "id": entry.id,
            "entry_number": entry.entry_number,
            "entry_type": entry.entry_type.value,
            "description": entry.description,
            "posted_at": entry.posted_at.isoformat(),
            "lines_count": len(entry.lines)
        },
        message="Journal entry posted successfully"
    )


@router.post("/entries/{entry_id}/reverse", response_model=StandardResponse[Dict[str, Any]])
def reverse_journal_entry(
    entry_id: str,
    reason: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Reverses an existing journal entry by posting an exact offsetting entry.
    Restricted to Platform Admin and Finance Operator.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance operators can reverse journal entries.")

    reversal = LedgerService.reverse_journal_entry(
        db=db,
        entry_id=entry_id,
        reason=reason,
        actor_id=current_user.id
    )

    return StandardResponse(
        data={
            "reversal_entry_id": reversal.id,
            "reversal_entry_number": reversal.entry_number,
            "reversed_original_id": entry_id
        },
        message="Journal entry reversed successfully"
    )
