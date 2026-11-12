"""
disbursements.py - Milestone Disbursement & Maker-Checker Payout API Endpoints
=============================================================================
Provides formal endpoints for:
1. Multi-gate eligibility evaluation
2. Maker request creation (Finance Operator)
3. Checker authorization and segregation-of-duties enforcement (Finance Approver)
4. Batch payout execution to PayPal
5. Status monitoring and audit trail inspection
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.api.schemas import (
    StandardResponse,
    DisbursementEligibilityResponse,
    DisbursementRequestPayload,
    DisbursementApprovalPayload,
    DisbursementRejectPayload,
    DisbursementResponse,
    PayoutBatchCreatePayload,
    PayoutBatchResponseSchema
)
from backend.app.core.constants import PlatformRole, OrganizationRole, DisbursementStatus
from backend.app.core.exceptions import ForbiddenException, NotFoundException
from backend.app.core.permissions import get_current_user, require_platform_role, require_org_role
from backend.app.db.session import get_db
from backend.app.db.models.users import User
from backend.app.db.models.financial import Disbursement, PayoutBatch
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.disbursement_eligibility_service import DisbursementEligibilityService

router = APIRouter(prefix="/disbursements", tags=["Disbursements & Payouts"])
disbursement_service = DisbursementService()


@router.get("/milestones/{milestone_id}/eligibility", response_model=StandardResponse[DisbursementEligibilityResponse])
def check_milestone_eligibility(
    milestone_id: str,
    requested_amount_minor_units: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Evaluates whether a milestone is eligible for fund disbursement across all 9 gates.
    """
    report = DisbursementEligibilityService.evaluate_eligibility(
        db=db,
        milestone_id=milestone_id,
        requested_amount_minor_units=requested_amount_minor_units
    )

    return StandardResponse(
        data=DisbursementEligibilityResponse(**report),
        message="Eligibility evaluation completed"
    )


@router.post("/request", response_model=StandardResponse[DisbursementResponse], status_code=status.HTTP_201_CREATED)
def request_milestone_disbursement(
    payload: DisbursementRequestPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Maker Step: Initiates a milestone disbursement request.
    Restricted to Platform Admin or Finance Operator.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_OPERATOR)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance operators can initiate disbursement requests.")

    disb = disbursement_service.request_disbursement(
        db=db,
        milestone_id=payload.milestone_id,
        amount_minor_units=payload.amount_minor_units,
        requester_id=current_user.id,
        recipient_id=payload.recipient_id,
        notes=payload.notes
    )

    return StandardResponse(
        data=DisbursementResponse(
            id=disb.id,
            milestone_id=disb.milestone_id,
            organization_id=disb.organization_id,
            review_case_id=disb.review_case_id,
            recipient_id=disb.recipient_id,
            payout_batch_id=disb.payout_batch_id,
            provider=disb.provider,
            provider_item_id=disb.provider_item_id,
            amount_minor_units=disb.amount_minor_units,
            gross_amount_minor_units=disb.gross_amount_minor_units,
            provider_fee_minor_units=disb.provider_fee_minor_units,
            net_amount_minor_units=disb.net_amount_minor_units,
            currency=disb.currency,
            status=disb.status.value,
            requested_by_id=disb.requested_by_id,
            approved_by_id=disb.approved_by_id,
            requested_at=disb.requested_at,
            approved_at=disb.approved_at,
            submitted_at=disb.submitted_at,
            disbursed_at=disb.disbursed_at,
            failure_reason=disb.failure_reason,
            created_at=disb.created_at
        ),
        message="Disbursement request submitted for checker authorization"
    )


@router.post("/{disbursement_id}/approve", response_model=StandardResponse[DisbursementResponse])
def approve_milestone_disbursement(
    disbursement_id: str,
    payload: Optional[DisbursementApprovalPayload] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Checker Step: Authorizes a milestone disbursement.
    Enforces strict segregation of duties: Requester cannot approve their own disbursement.
    Posts double-entry accounting commitment.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_APPROVER)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance approvers can authorize disbursements.")

    notes = payload.notes if payload else None
    disb = disbursement_service.approve_disbursement(
        db=db,
        disbursement_id=disbursement_id,
        approver_id=current_user.id,
        notes=notes
    )

    return StandardResponse(
        data=DisbursementResponse(
            id=disb.id,
            milestone_id=disb.milestone_id,
            organization_id=disb.organization_id,
            review_case_id=disb.review_case_id,
            recipient_id=disb.recipient_id,
            payout_batch_id=disb.payout_batch_id,
            provider=disb.provider,
            provider_item_id=disb.provider_item_id,
            amount_minor_units=disb.amount_minor_units,
            gross_amount_minor_units=disb.gross_amount_minor_units,
            provider_fee_minor_units=disb.provider_fee_minor_units,
            net_amount_minor_units=disb.net_amount_minor_units,
            currency=disb.currency,
            status=disb.status.value,
            requested_by_id=disb.requested_by_id,
            approved_by_id=disb.approved_by_id,
            requested_at=disb.requested_at,
            approved_at=disb.approved_at,
            submitted_at=disb.submitted_at,
            disbursed_at=disb.disbursed_at,
            failure_reason=disb.failure_reason,
            created_at=disb.created_at
        ),
        message="Disbursement authorized and committed to ledger"
    )


@router.post("/{disbursement_id}/reject", response_model=StandardResponse[DisbursementResponse])
def reject_milestone_disbursement(
    disbursement_id: str,
    payload: DisbursementRejectPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Checker Step: Rejects a disbursement request."""
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_APPROVER)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance approvers can reject disbursements.")

    disb = disbursement_service.reject_disbursement(
        db=db,
        disbursement_id=disbursement_id,
        rejecter_id=current_user.id,
        reason=payload.reason
    )

    return StandardResponse(
        data=DisbursementResponse(
            id=disb.id,
            milestone_id=disb.milestone_id,
            organization_id=disb.organization_id,
            review_case_id=disb.review_case_id,
            recipient_id=disb.recipient_id,
            payout_batch_id=disb.payout_batch_id,
            provider=disb.provider,
            provider_item_id=disb.provider_item_id,
            amount_minor_units=disb.amount_minor_units,
            gross_amount_minor_units=disb.gross_amount_minor_units,
            provider_fee_minor_units=disb.provider_fee_minor_units,
            net_amount_minor_units=disb.net_amount_minor_units,
            currency=disb.currency,
            status=disb.status.value,
            requested_by_id=disb.requested_by_id,
            approved_by_id=disb.approved_by_id,
            requested_at=disb.requested_at,
            approved_at=disb.approved_at,
            submitted_at=disb.submitted_at,
            disbursed_at=disb.disbursed_at,
            failure_reason=disb.failure_reason,
            created_at=disb.created_at
        ),
        message="Disbursement rejected"
    )


@router.post("/batches", response_model=StandardResponse[PayoutBatchResponseSchema], status_code=status.HTTP_201_CREATED)
def submit_payout_batch(
    payload: PayoutBatchCreatePayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Groups authorized disbursements into an outbound batch and submits to PayPal Payouts.
    Restricted to Platform Admin or Finance Approver.
    """
    allowed_roles = (PlatformRole.PLATFORM_ADMIN, PlatformRole.FINANCE_APPROVER)
    if current_user.platform_role not in allowed_roles:
        raise ForbiddenException("Only platform administrators and finance approvers can submit payout batches.")

    batch = disbursement_service.create_and_submit_payout_batch(
        db=db,
        disbursement_ids=payload.disbursement_ids,
        submitter_id=current_user.id,
        email_subject=payload.email_subject
    )

    return StandardResponse(
        data=PayoutBatchResponseSchema(
            id=batch.id,
            batch_reference_id=batch.batch_reference_id,
            provider=batch.provider,
            provider_batch_id=batch.provider_batch_id,
            status=batch.status.value,
            currency=batch.currency,
            total_items_count=batch.total_items_count,
            total_amount_minor_units=batch.total_amount_minor_units,
            total_fees_minor_units=batch.total_fees_minor_units,
            submitted_by_id=batch.submitted_by_id,
            submitted_at=batch.submitted_at,
            completed_at=batch.completed_at,
            created_at=batch.created_at
        ),
        message=f"Payout batch submitted with status: {batch.status.value}"
    )


@router.get("/batches/{batch_id}", response_model=StandardResponse[PayoutBatchResponseSchema])
def get_payout_batch(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves payout batch status and metrics."""
    batch = db.execute(
        select(PayoutBatch).where(
            (PayoutBatch.id == batch_id) | (PayoutBatch.batch_reference_id == batch_id)
        )
    ).scalar_one_or_none()

    if not batch:
        raise NotFoundException(f"Payout batch '{batch_id}' not found.")

    return StandardResponse(
        data=PayoutBatchResponseSchema(
            id=batch.id,
            batch_reference_id=batch.batch_reference_id,
            provider=batch.provider,
            provider_batch_id=batch.provider_batch_id,
            status=batch.status.value,
            currency=batch.currency,
            total_items_count=batch.total_items_count,
            total_amount_minor_units=batch.total_amount_minor_units,
            total_fees_minor_units=batch.total_fees_minor_units,
            submitted_by_id=batch.submitted_by_id,
            submitted_at=batch.submitted_at,
            completed_at=batch.completed_at,
            created_at=batch.created_at
        )
    )


@router.get("/{disbursement_id}", response_model=StandardResponse[DisbursementResponse])
def get_disbursement(
    disbursement_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Inspects a single disbursement record."""
    disb = db.execute(select(Disbursement).where(Disbursement.id == disbursement_id)).scalar_one_or_none()
    if not disb:
        raise NotFoundException(f"Disbursement '{disbursement_id}' not found.")

    return StandardResponse(
        data=DisbursementResponse(
            id=disb.id,
            milestone_id=disb.milestone_id,
            organization_id=disb.organization_id,
            review_case_id=disb.review_case_id,
            recipient_id=disb.recipient_id,
            payout_batch_id=disb.payout_batch_id,
            provider=disb.provider,
            provider_item_id=disb.provider_item_id,
            amount_minor_units=disb.amount_minor_units,
            gross_amount_minor_units=disb.gross_amount_minor_units,
            provider_fee_minor_units=disb.provider_fee_minor_units,
            net_amount_minor_units=disb.net_amount_minor_units,
            currency=disb.currency,
            status=disb.status.value,
            requested_by_id=disb.requested_by_id,
            approved_by_id=disb.approved_by_id,
            requested_at=disb.requested_at,
            approved_at=disb.approved_at,
            submitted_at=disb.submitted_at,
            disbursed_at=disb.disbursed_at,
            failure_reason=disb.failure_reason,
            created_at=disb.created_at
        )
    )


@router.get("/organizations/{org_id}", response_model=StandardResponse[List[DisbursementResponse]])
def get_organization_disbursements(
    org_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all disbursements for an organization."""
    if current_user.platform_role != PlatformRole.PLATFORM_ADMIN:
        require_org_role(org_id, current_user.id, db, [OrganizationRole.OWNER, OrganizationRole.ADMIN, OrganizationRole.MEMBER])

    disbursements = db.execute(
        select(Disbursement)
        .where(Disbursement.organization_id == org_id)
        .order_by(Disbursement.created_at.desc())
    ).scalars().all()

    return StandardResponse(
        data=[
            DisbursementResponse(
                id=d.id,
                milestone_id=d.milestone_id,
                organization_id=d.organization_id,
                review_case_id=d.review_case_id,
                recipient_id=d.recipient_id,
                payout_batch_id=d.payout_batch_id,
                provider=d.provider,
                provider_item_id=d.provider_item_id,
                amount_minor_units=d.amount_minor_units,
                gross_amount_minor_units=d.gross_amount_minor_units,
                provider_fee_minor_units=d.provider_fee_minor_units,
                net_amount_minor_units=d.net_amount_minor_units,
                currency=d.currency,
                status=d.status.value,
                requested_by_id=d.requested_by_id,
                approved_by_id=d.approved_by_id,
                requested_at=d.requested_at,
                approved_at=d.approved_at,
                submitted_at=d.submitted_at,
                disbursed_at=d.disbursed_at,
                failure_reason=d.failure_reason,
                created_at=d.created_at
            )
            for d in disbursements
        ]
    )

