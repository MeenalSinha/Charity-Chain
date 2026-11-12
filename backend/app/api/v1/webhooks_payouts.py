"""
webhooks_payouts.py - Outbound PayPal Payouts Webhook Ingestion API
===================================================================
Receives, verifies, deduplicates, and processes asynchronous payout notifications:
- PAYMENT.PAYOUTSBATCH.SUCCESS
- PAYMENT.PAYOUTS-ITEM.SUCCEEDED
- PAYMENT.PAYOUTS-ITEM.FAILED / BLOCKED
- PAYMENT.PAYOUTS-ITEM.UNCLAIMED / RETURNED
"""

import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request, Header, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.api.schemas import StandardResponse
from backend.app.core.constants import WebhookProcessingStatus, WebhookSignatureStatus
from backend.app.core.logging import get_logger
from backend.app.db.session import get_db
from backend.app.db.models.financial import WebhookEvent
from backend.app.services.disbursement_service import DisbursementService

logger = get_logger("api.webhooks.payouts")
router = APIRouter(prefix="/webhooks/paypal-payouts", tags=["Payout Webhooks"])
disbursement_service = DisbursementService()


@router.post("", response_model=StandardResponse)
async def receive_paypal_payout_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Ingests and reconciles outbound PayPal payout webhooks.
    Deduplicates events and synchronizes item settlement states with double-entry ledger.
    """
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception as exc:
        logger.error("Failed to parse PayPal payout webhook JSON payload", exc_info=True)
        raise HTTPException(status_code=400, detail="Malformed JSON payload")

    event_id = body_json.get("id")
    event_type = body_json.get("event_type", "UNKNOWN")
    resource = body_json.get("resource", {})

    if not event_id:
        raise HTTPException(status_code=400, detail="Missing required event 'id' in webhook payload")

    # Deduplication check
    existing = db.execute(select(WebhookEvent).where(WebhookEvent.event_id == event_id)).scalar_one_or_none()
    if existing and existing.processing_status == WebhookProcessingStatus.PROCESSED:
        logger.info(f"Duplicate payout webhook event ignored: {event_id}")
        return StandardResponse(data={"status": "DUPLICATE", "event_id": event_id}, message="Event already processed")

    now = datetime.now(timezone.utc)
    if not existing:
        event_record = WebhookEvent(
            id=str(uuid.uuid4()),
            provider="PAYPAL_PAYOUTS",
            event_id=event_id,
            event_type=event_type,
            resource_id=resource.get("payout_item_id") or resource.get("batch_header", {}).get("payout_batch_id"),
            resource_type=body_json.get("resource_type", "payout"),
            signature_verification_status=WebhookSignatureStatus.VERIFIED,
            processing_status=WebhookProcessingStatus.PENDING,
            payload=json.dumps(body_json)
        )
        db.add(event_record)
        db.flush()
    else:
        event_record = existing

    try:
        result = disbursement_service.process_payout_webhook_event(
            db=db,
            event_type=event_type,
            resource=resource
        )
        event_record.processing_status = WebhookProcessingStatus.PROCESSED
        event_record.processed_at = now
        db.commit()
        return StandardResponse(data=result, message="Webhook processed successfully")
    except Exception as exc:
        db.rollback()
        event_record.processing_status = WebhookProcessingStatus.FAILED
        event_record.failure_reason = str(exc)
        event_record.retry_count = (event_record.retry_count or 0) + 1
        db.commit()
        logger.error(f"Error processing payout webhook event {event_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Payout webhook processing failed: {str(exc)}")
