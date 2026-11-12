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
from backend.app.core.exceptions import ValidationException
from backend.app.integrations.payments.exceptions import WebhookVerificationFailedException
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
    Cryptographically verifies signatures, deduplicates events, and synchronizes item settlement states.
    """
    try:
        raw_body = await request.body()
    except Exception as exc:
        logger.error("Failed to read payout webhook request body", exc_info=True)
        raise HTTPException(status_code=400, detail="Malformed request body")

    headers = dict(request.headers)

    try:
        result = disbursement_service.process_paypal_payout_webhook(
            db=db,
            headers=headers,
            raw_body=raw_body
        )
        return StandardResponse(data=result, message="Webhook processed successfully")
    except WebhookVerificationFailedException as exc:
        logger.warning(f"Rejected unverified PayPal payout webhook: {exc.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed"
        )
    except ValidationException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error processing PayPal payout webhook: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Payout webhook processing failed: {str(exc)}"
        )


