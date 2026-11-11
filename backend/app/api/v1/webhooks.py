"""
webhooks.py - PayPal Webhook Notification Endpoint
===================================================
Receives asynchronous PayPal events, cryptographically verifies signatures, and reconciles records.
"""

from fastapi import APIRouter, Request, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.logging import get_logger
from backend.app.db.session import get_db
from backend.app.services.donation_service import DonationService
from backend.app.integrations.payments.exceptions import WebhookVerificationFailedException

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])
logger = get_logger("api.v1.webhooks")
donation_service = DonationService()


@router.post("/paypal", status_code=status.HTTP_200_OK)
async def receive_paypal_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Ingests and processes PayPal webhook notifications.
    Verifies cryptographic signatures, deduplicates events, and triggers payment reconciliations.
    """
    raw_body = await request.body()
    headers = dict(request.headers)

    try:
        res = donation_service.process_paypal_webhook(
            db=db,
            headers=headers,
            raw_body=raw_body
        )
        return res
    except WebhookVerificationFailedException as exc:
        logger.warning(f"Rejected unverified PayPal webhook: {exc.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed"
        )
    except Exception as exc:
        logger.error("Error processing PayPal webhook", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing error: {str(exc)}"
        )
