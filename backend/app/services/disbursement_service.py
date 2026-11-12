"""
disbursement_service.py - Milestone Disbursement & Outbound Payout Orchestration
=================================================================================
Orchestrates genuine outbound payouts with strict financial controls:
- Maker-Checker segregation: Requester (Operator) != Approver (Checker). Self-approval forbidden.
- Eligibility pre-validation: All 9 compliance and verification gates must pass.
- Double-entry ledger integration: Automatic commitment and settlement journal entries.
- Provider abstraction: Genuine PayPal Payouts with mock injection support.
- Asynchronous batch execution, webhook handling, and failure reversals.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    DisbursementStatus, PlatformRole, RecipientType, DisbursementMethod
)
from backend.app.core.exceptions import (
    ValidationException, NotFoundException, ForbiddenException, ConflictException
)
from backend.app.core.logging import get_logger
from backend.app.db.models.financial import (
    Disbursement, PayoutRecipient, PayoutBatch, PayoutApproval
)
from backend.app.db.models.projects import Milestone
from backend.app.db.models.users import User
from backend.app.integrations.payouts.base import PayoutProviderInterface
from backend.app.integrations.payouts.factory import get_payout_provider
from backend.app.integrations.payouts.schemas import (
    PayoutBatchRequest, PayoutItemRequest
)
from backend.app.services.recipient_service import PayoutRecipientService
from backend.app.services.fund_allocation_service import FundAllocationService
from backend.app.services.ledger_service import LedgerService
from backend.app.services.disbursement_eligibility_service import DisbursementEligibilityService

logger = get_logger("services.disbursement")


class DisbursementService:
    """Core domain service for milestone fund disbursements and maker-checker payouts."""

    def __init__(self, provider: Optional[PayoutProviderInterface] = None):
        self.provider = provider or get_payout_provider()
        self.recipient_service = PayoutRecipientService(provider=self.provider)

    # -------------------------------------------------------------------------
    # 1. Maker Step: Create Disbursement Request
    # -------------------------------------------------------------------------

    def request_disbursement(
        self,
        db: Session,
        milestone_id: str,
        amount_minor_units: int,
        requester_id: str,
        recipient_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Disbursement:
        """
        Maker step: A finance operator requests a milestone disbursement.
        Pre-validates eligibility and available funds.
        Status: PENDING_APPROVAL.
        """
        if amount_minor_units <= 0:
            raise ValidationException("Disbursement amount must be greater than zero.")

        # Evaluate complete eligibility gate
        report = DisbursementEligibilityService.evaluate_eligibility(
            db=db,
            milestone_id=milestone_id,
            requested_amount_minor_units=amount_minor_units
        )

        if not report["is_eligible"]:
            reasons = "; ".join(report["rejection_reasons"])
            raise ValidationException(f"Milestone is not eligible for disbursement: {reasons}")

        # Resolve payout recipient
        final_recipient_id = recipient_id or report["recipient_id"]
        if not final_recipient_id:
            raise ValidationException("No verified payout recipient available for this organization.")

        requester = db.execute(select(User).where(User.id == requester_id)).scalar_one_or_none()
        requester_role = requester.platform_role.value if requester else "FINANCE_OPERATOR"

        now = datetime.now(timezone.utc)
        disbursement = Disbursement(
            id=str(uuid.uuid4()),
            milestone_id=milestone_id,
            organization_id=report["organization_id"],
            review_case_id=report["review_case_id"],
            recipient_id=final_recipient_id,
            amount_minor_units=amount_minor_units,
            gross_amount_minor_units=amount_minor_units,
            net_amount_minor_units=amount_minor_units,
            currency=report["currency"],
            status=DisbursementStatus.PENDING_APPROVAL,
            requested_by_id=requester_id,
            requested_at=now,
            idempotency_key=f"DISB-{milestone_id}-{str(uuid.uuid4())[:8].upper()}"
        )
        db.add(disbursement)
        db.flush()

        # Record Maker audit action
        approval_log = PayoutApproval(
            id=str(uuid.uuid4()),
            disbursement_id=disbursement.id,
            action="REQUESTED",
            actor_id=requester_id,
            actor_role=requester_role,
            notes=notes or "Milestone disbursement initiated",
            decided_at=now
        )
        db.add(approval_log)
        db.commit()
        db.refresh(disbursement)

        logger.info(
            f"Disbursement {disbursement.id} requested by {requester_id} for {amount_minor_units} minor units",
            extra={"disbursement_id": disbursement.id, "requester_id": requester_id}
        )
        return disbursement

    # -------------------------------------------------------------------------
    # 2. Checker Step: Approve or Reject
    # -------------------------------------------------------------------------

    def approve_disbursement(
        self,
        db: Session,
        disbursement_id: str,
        approver_id: str,
        notes: Optional[str] = None
    ) -> Disbursement:
        """
        Checker step: An authorized finance approver verifies and authorizes payout.
        Enforces strict segregation of duties: Requester cannot approve their own disbursement.
        Posts Double-Entry Ledger commitment upon approval.
        """
        disbursement = db.execute(select(Disbursement).where(Disbursement.id == disbursement_id)).scalar_one_or_none()
        if not disbursement:
            raise NotFoundException(f"Disbursement '{disbursement_id}' not found.")

        if disbursement.status not in (DisbursementStatus.REQUESTED, DisbursementStatus.PENDING_APPROVAL):
            raise ValidationException(f"Cannot approve disbursement in status '{disbursement.status.value}'.")

        # Segregation of Duties Check
        if disbursement.requested_by_id == approver_id:
            raise ForbiddenException(
                "Maker-Checker financial control violation: You cannot approve a disbursement request you initiated. An independent approver is required."
            )

        approver = db.execute(select(User).where(User.id == approver_id)).scalar_one_or_none()
        approver_role = approver.platform_role.value if approver else "FINANCE_APPROVER"

        # Re-check funds
        fund_summary = FundAllocationService.get_milestone_fund_summary(db, disbursement.milestone_id)
        if disbursement.amount_minor_units > fund_summary["available_balance_minor_units"]:
            raise ValidationException("Insufficient unencumbered funds available to approve this disbursement.")

        now = datetime.now(timezone.utc)
        disbursement.status = DisbursementStatus.APPROVED
        disbursement.approved_by_id = approver_id
        disbursement.approved_at = now

        approval_log = PayoutApproval(
            id=str(uuid.uuid4()),
            disbursement_id=disbursement.id,
            action="APPROVED",
            actor_id=approver_id,
            actor_role=approver_role,
            notes=notes or "Disbursement authorized by finance checker",
            decided_at=now
        )
        db.add(approval_log)

        # Post double-entry accounting commitment
        LedgerService.record_payout_commitment(db=db, disbursement=disbursement, actor_id=approver_id)

        db.commit()
        db.refresh(disbursement)

        logger.info(
            f"Disbursement {disbursement.id} approved by checker {approver_id}",
            extra={"disbursement_id": disbursement.id, "approver_id": approver_id}
        )
        return disbursement

    def reject_disbursement(
        self,
        db: Session,
        disbursement_id: str,
        rejecter_id: str,
        reason: str
    ) -> Disbursement:
        """Checker step: Rejects disbursement request."""
        disbursement = db.execute(select(Disbursement).where(Disbursement.id == disbursement_id)).scalar_one_or_none()
        if not disbursement:
            raise NotFoundException(f"Disbursement '{disbursement_id}' not found.")

        if disbursement.status not in (DisbursementStatus.REQUESTED, DisbursementStatus.PENDING_APPROVAL):
            raise ValidationException(f"Cannot reject disbursement in status '{disbursement.status.value}'.")

        rejecter = db.execute(select(User).where(User.id == rejecter_id)).scalar_one_or_none()
        rejecter_role = rejecter.platform_role.value if rejecter else "FINANCE_APPROVER"

        now = datetime.now(timezone.utc)
        disbursement.status = DisbursementStatus.CANCELLED
        disbursement.failure_reason = reason

        approval_log = PayoutApproval(
            id=str(uuid.uuid4()),
            disbursement_id=disbursement.id,
            action="REJECTED",
            actor_id=rejecter_id,
            actor_role=rejecter_role,
            notes=reason,
            decided_at=now
        )
        db.add(approval_log)
        db.commit()
        db.refresh(disbursement)
        return disbursement

    # -------------------------------------------------------------------------
    # 3. Batch Submission to Payment Provider
    # -------------------------------------------------------------------------

    def create_and_submit_payout_batch(
        self,
        db: Session,
        disbursement_ids: List[str],
        submitter_id: str,
        email_subject: Optional[str] = None
    ) -> PayoutBatch:
        """
        Groups approved disbursements into a batch, submits to payout provider (PayPal Payouts),
        and updates disbursement statuses.
        """
        if not disbursement_ids:
            raise ValidationException("At least one disbursement ID must be provided to build a batch.")

        disbursements = db.execute(
            select(Disbursement).where(Disbursement.id.in_(disbursement_ids))
        ).scalars().all()

        if len(disbursements) != len(disbursement_ids):
            raise NotFoundException("One or more disbursement records could not be found.")

        # Ensure all are in APPROVED status
        for d in disbursements:
            if d.status != DisbursementStatus.APPROVED:
                raise ValidationException(
                    f"Disbursement {d.id} is in status '{d.status.value}'. Only APPROVED disbursements can be submitted."
                )

        # Build items for payout provider
        items_requests = []
        total_minor = 0
        currency = disbursements[0].currency

        for d in disbursements:
            total_minor += d.amount_minor_units
            receiver_dest = self.recipient_service.get_decrypted_destination_for_payout(db, d.recipient_id)
            items_requests.append(
                PayoutItemRequest(
                    disbursement_id=d.id,
                    recipient_type=RecipientType.PAYPAL_EMAIL,
                    receiver=receiver_dest,
                    amount_minor_units=d.amount_minor_units,
                    currency=d.currency,
                    note=f"CharityChain milestone disbursement {d.milestone_id}",
                    sender_item_id=d.id
                )
            )

        now = datetime.now(timezone.utc)
        sender_batch_id = f"BATCH-{now.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

        batch_req = PayoutBatchRequest(
            sender_batch_id=sender_batch_id,
            email_subject=email_subject or "CharityChain milestone disbursement notice",
            items=items_requests
        )

        # Persist local PayoutBatch
        batch = PayoutBatch(
            id=str(uuid.uuid4()),
            batch_reference_id=sender_batch_id,
            provider="PAYPAL_PAYOUTS",
            status=DisbursementStatus.SUBMITTED,
            currency=currency,
            total_items_count=len(disbursements),
            total_amount_minor_units=total_minor,
            submitted_by_id=submitter_id,
            submitted_at=now
        )
        db.add(batch)
        db.flush()

        # Link disbursements to batch
        for d in disbursements:
            d.payout_batch_id = batch.id
            d.status = DisbursementStatus.SUBMITTED
            d.submitted_at = now

        db.commit()

        # Submit to external gateway
        try:
            gateway_res = self.provider.create_payout_batch(batch_req)
        except Exception as exc:
            logger.error(f"Failed to submit payout batch {sender_batch_id}", exc_info=True)
            batch.status = DisbursementStatus.FAILED
            for d in disbursements:
                d.status = DisbursementStatus.FAILED
                d.failure_reason = f"Provider submission failure: {str(exc)}"
                # Reverse ledger commitment back to designated funds
                LedgerService.record_payout_failure_reversal(db, d, actor_id=submitter_id)
            db.commit()
            raise exc

        # Update batch with provider details
        batch.provider_batch_id = gateway_res.payout_batch_id
        batch.total_fees_minor_units = gateway_res.total_fee_minor_units

        # Map item responses back to disbursements
        res_item_map = {it.sender_item_id: it for it in gateway_res.items}

        all_succeeded = True
        for d in disbursements:
            item_res = res_item_map.get(d.id)
            if item_res:
                d.provider_item_id = item_res.payout_item_id
                d.provider_batch_id = gateway_res.payout_batch_id
                d.provider_fee_minor_units = item_res.fee_minor_units
                d.net_amount_minor_units = d.amount_minor_units

                if item_res.transaction_status in ("SUCCESS", "COMPLETED"):
                    d.status = DisbursementStatus.SUCCEEDED
                    d.disbursed_at = now
                    # Post Double-Entry Execution Entry
                    LedgerService.record_payout_execution(
                        db=db,
                        disbursement=d,
                        provider_fee_minor_units=item_res.fee_minor_units,
                        actor_id=submitter_id
                    )
                elif item_res.transaction_status in ("FAILED", "BLOCKED"):
                    d.status = DisbursementStatus.FAILED
                    d.failure_reason = str(item_res.errors or "Provider transaction declined")
                    all_succeeded = False
                    LedgerService.record_payout_failure_reversal(db=db, disbursement=d, actor_id=submitter_id)
                else:
                    d.status = DisbursementStatus.PROCESSING
                    all_succeeded = False
            else:
                d.status = DisbursementStatus.PROCESSING
                all_succeeded = False

        if all_succeeded:
            batch.status = DisbursementStatus.SUCCEEDED
            batch.completed_at = now
        else:
            batch.status = DisbursementStatus.PROCESSING

        db.commit()
        db.refresh(batch)

        logger.info(
            f"Payout batch {batch.id} submitted (Provider Batch ID: {batch.provider_batch_id}, Status: {batch.status.value})"
        )
        return batch

    # -------------------------------------------------------------------------
    # 4. Webhook Event Processing
    # -------------------------------------------------------------------------

    def process_payout_webhook_event(
        self,
        db: Session,
        event_type: str,
        resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes incoming PayPal Payouts webhook events idempotently.
        Handles:
        - PAYMENT.PAYOUTSBATCH.SUCCESS
        - PAYMENT.PAYOUTS-ITEM.SUCCEEDED
        - PAYMENT.PAYOUTS-ITEM.FAILED / BLOCKED
        - PAYMENT.PAYOUTS-ITEM.RETURNED / UNCLAIMED
        """
        logger.info(f"Processing payout webhook event: {event_type}")
        now = datetime.now(timezone.utc)

        if event_type == "PAYMENT.PAYOUTSBATCH.SUCCESS":
            payout_batch_id = resource.get("batch_header", {}).get("payout_batch_id")
            if payout_batch_id:
                batch = db.execute(
                    select(PayoutBatch).where(PayoutBatch.provider_batch_id == payout_batch_id)
                ).scalar_one_or_none()
                if batch:
                    batch.status = DisbursementStatus.SUCCEEDED
                    batch.completed_at = now
                    db.commit()
            return {"status": "SUCCESS", "event_type": event_type}

        elif event_type in ("PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "PAYMENT.PAYOUTS-ITEM.COMPLETED"):
            payout_item_id = resource.get("payout_item_id")
            sender_item_id = resource.get("payout_item", {}).get("sender_item_id")

            disbursement = None
            if sender_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.id == sender_item_id)).scalar_one_or_none()
            if not disbursement and payout_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.provider_item_id == payout_item_id)).scalar_one_or_none()

            if disbursement and disbursement.status != DisbursementStatus.SUCCEEDED:
                disbursement.status = DisbursementStatus.SUCCEEDED
                disbursement.disbursed_at = now
                if payout_item_id:
                    disbursement.provider_item_id = payout_item_id

                # Fee calculation
                fee_obj = resource.get("payout_item_fee", {})
                fee_val = fee_obj.get("value", "0.00")
                fee_minor = int(float(fee_val) * 100) if fee_val else 0
                disbursement.provider_fee_minor_units = fee_minor

                LedgerService.record_payout_execution(
                    db=db,
                    disbursement=disbursement,
                    provider_fee_minor_units=fee_minor
                )
                db.commit()

            return {"status": "SUCCESS", "disbursement_id": disbursement.id if disbursement else None}

        elif event_type in ("PAYMENT.PAYOUTS-ITEM.FAILED", "PAYMENT.PAYOUTS-ITEM.BLOCKED", "PAYMENT.PAYOUTS-ITEM.DENIED"):
            payout_item_id = resource.get("payout_item_id")
            sender_item_id = resource.get("payout_item", {}).get("sender_item_id")

            disbursement = None
            if sender_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.id == sender_item_id)).scalar_one_or_none()
            if not disbursement and payout_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.provider_item_id == payout_item_id)).scalar_one_or_none()

            if disbursement and disbursement.status != DisbursementStatus.FAILED:
                disbursement.status = DisbursementStatus.FAILED
                errors = resource.get("errors", {})
                disbursement.failure_reason = str(errors or "Payout item rejected by provider")
                LedgerService.record_payout_failure_reversal(db=db, disbursement=disbursement)
                db.commit()

            return {"status": "FAILED_RECORDED", "disbursement_id": disbursement.id if disbursement else None}

        elif event_type in ("PAYMENT.PAYOUTS-ITEM.UNCLAIMED", "PAYMENT.PAYOUTS-ITEM.RETURNED"):
            payout_item_id = resource.get("payout_item_id")
            sender_item_id = resource.get("payout_item", {}).get("sender_item_id")

            disbursement = None
            if sender_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.id == sender_item_id)).scalar_one_or_none()
            if not disbursement and payout_item_id:
                disbursement = db.execute(select(Disbursement).where(Disbursement.provider_item_id == payout_item_id)).scalar_one_or_none()

            if disbursement and disbursement.status != DisbursementStatus.RETURNED:
                disbursement.status = DisbursementStatus.RETURNED
                disbursement.failure_reason = f"Funds unclaimed or returned by provider: {event_type}"
                LedgerService.record_payout_failure_reversal(db=db, disbursement=disbursement)
                db.commit()

            return {"status": "RETURNED_RECORDED", "disbursement_id": disbursement.id if disbursement else None}

        return {"status": "IGNORED", "event_type": event_type}

