"""
donation_service.py - Core Charitable Donation & PayPal Payment Processing Service
===================================================================================
Orchestrates genuine PayPal Orders v2 transactions:
1. Validates project, milestone, and NGO eligibility
2. Creates Donation intent and registers PayPal order
3. Captures approved payments idempotently
4. Reconciles webhook notifications with local records
5. Handles partial and full refunds with funding aggregate adjustment
6. Ensures financial integrity: integer minor units, no fake escrow, no premature disbursement
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.constants import (
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus,
    PaymentRefundStatus, ProjectStatus, MilestoneStatus,
    OrganizationStatus, ProjectVisibility, WebhookProcessingStatus,
    WebhookSignatureStatus, PlatformRole
)
from backend.app.core.currency import Money
from backend.app.core.exceptions import (
    ValidationException, NotFoundException, ForbiddenException,
    ConflictException
)
from backend.app.core.logging import get_logger
from backend.app.core.state_machines import StateMachineValidator
from backend.app.db.models.financial import (
    Donation, PaymentOrder, PaymentCapture, PaymentRefund, WebhookEvent
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import User, Organization
from backend.app.db.models.reviews_impact import AuditEvent
from backend.app.integrations.payments.base import PaymentProviderInterface
from backend.app.integrations.payments.factory import get_payment_provider
from backend.app.integrations.payments.exceptions import (
    PaymentException, PaymentProviderException, PaymentAuthenticationException,
    PaymentOrderNotFoundException, PaymentCaptureFailedException,
    PaymentRefundFailedException, WebhookVerificationFailedException,
    InvalidPaymentAmountException, CurrencyMismatchException
)

logger = get_logger("services.donation")


class DonationService:
    """Service governing charitable contributions, PayPal transactions, and financial integrity."""

    def __init__(self, provider: Optional[PaymentProviderInterface] = None):
        self._provider = provider

    @property
    def provider(self) -> PaymentProviderInterface:
        if self._provider is not None:
            return self._provider
        return get_payment_provider()

    # -------------------------------------------------------------------------
    # 1. Donation Intent & PayPal Order Creation
    # -------------------------------------------------------------------------

    def create_donation_intent(
        self,
        db: Session,
        project_id: str,
        milestone_id: str,
        amount_minor_units: int,
        currency: str = "USD",
        donor_id: Optional[str] = None,
        donor_name: Optional[str] = None,
        donor_email: Optional[str] = None,
        is_anonymous: bool = False,
        return_url: Optional[str] = None,
        cancel_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new Donation intent and generates an authentic PayPal Orders v2 checkout order.
        Validates all eligibility rules server-side before contacting the payment gateway.
        """
        curr = currency.strip().upper()
        if curr not in settings.SUPPORTED_CURRENCIES:
            raise ValidationException(f"Currency '{curr}' is not supported. Permitted: {settings.SUPPORTED_CURRENCIES}")

        if amount_minor_units < settings.MIN_DONATION_AMOUNT:
            min_money = Money(settings.MIN_DONATION_AMOUNT, curr)
            raise InvalidPaymentAmountException(f"Minimum donation amount is {min_money.minor_units} minor units ({min_money.to_major_units()} {curr})")

        if amount_minor_units > settings.MAX_DONATION_AMOUNT:
            max_money = Money(settings.MAX_DONATION_AMOUNT, curr)
            raise InvalidPaymentAmountException(f"Maximum donation amount is {max_money.minor_units} minor units ({max_money.to_major_units()} {curr})")

        # 1. Verify Project
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project:
            raise NotFoundException(f"Project with ID '{project_id}' not found")

        if project.is_archived:
            raise ValidationException("Cannot donate to an archived project")

        if project.status != ProjectStatus.ACTIVE:
            raise ValidationException(f"Project must be ACTIVE to accept contributions. Current status: {project.status.value}")

        if project.public_visibility != ProjectVisibility.PUBLIC:
            raise ValidationException("Project is not publicly visible for contributions")

        if project.currency.upper() != curr:
            raise CurrencyMismatchException(expected=project.currency.upper(), actual=curr)

        # 2. Verify Sponsoring Organization
        org = db.execute(select(Organization).where(Organization.id == project.organization_id)).scalar_one_or_none()
        if not org or org.status != OrganizationStatus.VERIFIED:
            raise ValidationException("Sponsoring organization is not verified. Donations are suspended.")

        # 3. Verify Milestone
        milestone = db.execute(select(Milestone).where(Milestone.id == milestone_id)).scalar_one_or_none()
        if not milestone:
            raise NotFoundException(f"Milestone with ID '{milestone_id}' not found")

        if milestone.project_id != project_id:
            raise ValidationException(f"Milestone '{milestone_id}' does not belong to Project '{project_id}'")

        if milestone.status in (MilestoneStatus.CANCELLED, MilestoneStatus.COMPLETED):
            raise ValidationException(f"Milestone is {milestone.status.value} and cannot accept new contributions")

        # 4. Resolve Donor User if authenticated
        donor_user = None
        if donor_id:
            donor_user = db.execute(select(User).where(User.id == donor_id)).scalar_one_or_none()
            if donor_user:
                if not donor_email:
                    donor_email = donor_user.email
                if not donor_name:
                    donor_name = f"{donor_user.profile.first_name} {donor_user.profile.last_name}".strip() if donor_user.profile else donor_user.email

        # 5. Create local Donation record in CREATED state
        donation_id = str(uuid.uuid4())
        donation = Donation(
            id=donation_id,
            donor_id=donor_id,
            project_id=project_id,
            milestone_id=milestone_id,
            amount_minor_units=amount_minor_units,
            currency=curr,
            status=DonationStatus.CREATED,
            donor_name=donor_name,
            donor_email=donor_email,
            is_anonymous=is_anonymous
        )
        db.add(donation)
        db.flush()

        # 6. Request genuine PayPal order
        money = Money(minor_units=amount_minor_units, currency=curr)
        ret_url = return_url or settings.FRONTEND_RETURN_URL
        cnc_url = cancel_url or settings.FRONTEND_CANCEL_URL

        try:
            paypal_res = self.provider.create_order(
                amount=money,
                reference_id=donation.id,
                return_url=ret_url,
                cancel_url=cnc_url,
                custom_id=donation.id
            )
        except PaymentException as exc:
            db.rollback()
            logger.error("PayPal order generation failed", exc_info=True)
            raise exc

        provider_order_id = paypal_res.get("provider_order_id") or paypal_res.get("id")
        approval_url = paypal_res.get("approval_url")

        # 7. Create PaymentOrder record
        payment_order = PaymentOrder(
            id=str(uuid.uuid4()),
            donation_id=donation.id,
            provider="PAYPAL",
            provider_order_id=provider_order_id,
            amount_minor_units=amount_minor_units,
            currency=curr,
            status=PaymentOrderStatus.CREATED,
            approval_url=approval_url,
            idempotency_key=str(uuid.uuid4())
        )
        db.add(payment_order)

        # Transition donation to PAYMENT_PENDING
        StateMachineValidator.validate_donation_transition(donation.status, DonationStatus.PAYMENT_PENDING)
        donation.status = DonationStatus.PAYMENT_PENDING
        db.commit()
        db.refresh(donation)
        db.refresh(payment_order)

        logger.info(
            "Donation intent and PayPal order created",
            extra={"donation_id": donation.id, "provider_order_id": provider_order_id, "amount": amount_minor_units, "currency": curr}
        )

        return {
            "donation_id": donation.id,
            "order_id": payment_order.id,
            "provider_order_id": provider_order_id,
            "approval_url": approval_url,
            "status": donation.status.value,
            "amount_minor_units": amount_minor_units,
            "currency": curr
        }

    # -------------------------------------------------------------------------
    # 2. PayPal Payment Capture
    # -------------------------------------------------------------------------

    def capture_donation_payment(
        self,
        db: Session,
        order_identifier: str,
        user_id: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Captures an approved PayPal order.
        Verifies that the gateway genuinely approved the transaction before persisting capture.
        Idempotent: if called multiple times, returns existing confirmed capture.
        """
        # 1. Resolve PaymentOrder by ID or provider_order_id
        order = db.execute(
            select(PaymentOrder).where(
                (PaymentOrder.id == order_identifier) | (PaymentOrder.provider_order_id == order_identifier)
            )
        ).scalar_one_or_none()

        if not order:
            raise PaymentOrderNotFoundException(f"Payment order '{order_identifier}' not found")

        donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
        if not donation:
            raise NotFoundException("Associated donation record not found")

        # 2. Check Idempotency: if already completed, return existing capture
        if order.status == PaymentOrderStatus.COMPLETED and donation.status == DonationStatus.PAID:
            existing_capture = db.execute(
                select(PaymentCapture).where(PaymentCapture.order_id == order.id)
            ).scalar_one_or_none()
            if existing_capture:
                logger.info(f"Order {order.id} is already captured. Returning existing capture record.")
                return {
                    "capture_id": existing_capture.id,
                    "order_id": order.id,
                    "provider_capture_id": existing_capture.provider_capture_id,
                    "status": existing_capture.status.value,
                    "amount_minor_units": existing_capture.amount_minor_units,
                    "currency": existing_capture.currency,
                    "donation_status": donation.status.value
                }

        # 3. Ownership security check (if user is authenticated and not admin)
        if donation.donor_id and user_id and donation.donor_id != user_id:
            # Check if user is platform admin
            user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
            if not user or user.platform_role != PlatformRole.PLATFORM_ADMIN:
                raise ForbiddenException("Access denied: You do not own this donation order.")

        # 4. Verify actual state at PayPal
        paypal_order = self.provider.get_order(order.provider_order_id)
        gateway_status = paypal_order.get("status")

        if gateway_status not in ("APPROVED", "COMPLETED"):
            raise PaymentCaptureFailedException(
                f"Cannot capture order: PayPal status is '{gateway_status}'. Order must be APPROVED by the donor."
            )

        # 5. Verify amounts match
        try:
            pu = paypal_order.get("purchase_units", [])[0]
            val_str = pu.get("amount", {}).get("value")
            curr_code = pu.get("amount", {}).get("currency_code")
            expected_money = Money(order.amount_minor_units, order.currency)
            expected_formatted = self.provider.format_money_value(expected_money)

            if curr_code != order.currency or val_str != expected_formatted:
                raise ValidationException(
                    f"Order amount mismatch with gateway: expected {expected_formatted} {order.currency}, got {val_str} {curr_code}"
                )
        except (IndexError, KeyError):
            pass

        # 6. Execute capture if not already COMPLETED at gateway
        capture_id_val = None
        if gateway_status == "COMPLETED":
            # Already captured at PayPal (e.g. captured via webhook or auto-capture)
            try:
                captures = paypal_order.get("purchase_units", [])[0].get("payments", {}).get("captures", [])
                if captures:
                    capture_id_val = captures[0].get("id")
            except Exception:
                pass
        else:
            capture_res = self.provider.capture_order(
                order.provider_order_id,
                idempotency_key=idempotency_key or order.idempotency_key
            )
            capture_id_val = capture_res.get("capture_id") or capture_res.get("id")

        if not capture_id_val:
            # Fallback retrieve capture ID from PayPal order
            fresh_order = self.provider.get_order(order.provider_order_id)
            try:
                captures = fresh_order.get("purchase_units", [])[0].get("payments", {}).get("captures", [])
                if captures:
                    capture_id_val = captures[0].get("id")
            except Exception:
                pass

        if not capture_id_val:
            raise PaymentCaptureFailedException(
                f"PayPal capture succeeded but no genuine provider capture ID was obtained for order {order.provider_order_id}"
            )

        # 7. Persist PaymentCapture record
        now_utc = datetime.now(timezone.utc)
        payment_capture = PaymentCapture(
            id=str(uuid.uuid4()),
            order_id=order.id,
            provider_capture_id=capture_id_val,
            amount_minor_units=order.amount_minor_units,
            currency=order.currency,
            status=PaymentCaptureStatus.COMPLETED,
            idempotency_key=idempotency_key or str(uuid.uuid4()),
            captured_at=now_utc
        )
        db.add(payment_capture)

        # 8. Update PaymentOrder and Donation states
        StateMachineValidator.validate_payment_order_transition(order.status, PaymentOrderStatus.COMPLETED)
        order.status = PaymentOrderStatus.COMPLETED

        StateMachineValidator.validate_donation_transition(donation.status, DonationStatus.PAID)
        donation.status = DonationStatus.PAID
        donation.completed_at = now_utc

        # 9. Update Project and Milestone funding totals
        project = db.execute(select(Project).where(Project.id == donation.project_id)).scalar_one_or_none()
        milestone = db.execute(select(Milestone).where(Milestone.id == donation.milestone_id)).scalar_one_or_none()

        if milestone:
            milestone.raised_amount_minor_units = (milestone.raised_amount_minor_units or 0) + order.amount_minor_units
        if project:
            project.raised_amount_minor_units = (project.raised_amount_minor_units or 0) + order.amount_minor_units

        # 10. Record Audit Event
        audit_event = AuditEvent(
            id=str(uuid.uuid4()),
            actor_id=user_id or donation.donor_id,
            action="DONATION_PAYMENT_CAPTURED",
            entity_type="Donation",
            entity_id=donation.id,
            ip_address="internal",
            details_json=json.dumps({
                "order_id": order.id,
                "provider_order_id": order.provider_order_id,
                "capture_id": payment_capture.id,
                "provider_capture_id": payment_capture.provider_capture_id,
                "amount_minor_units": order.amount_minor_units,
                "currency": order.currency,
                "project_id": donation.project_id,
                "milestone_id": donation.milestone_id
            })
        )
        db.add(audit_event)
        db.commit()
        db.refresh(payment_capture)

        logger.info(
            f"Successfully captured payment for Donation {donation.id}",
            extra={"order_id": order.id, "capture_id": payment_capture.id, "amount": order.amount_minor_units}
        )

        return {
            "capture_id": payment_capture.id,
            "order_id": order.id,
            "provider_capture_id": payment_capture.provider_capture_id,
            "status": payment_capture.status.value,
            "amount_minor_units": payment_capture.amount_minor_units,
            "currency": payment_capture.currency,
            "donation_status": donation.status.value
        }

    # -------------------------------------------------------------------------
    # 3. Cancellation Handling
    # -------------------------------------------------------------------------

    def cancel_donation(self, db: Session, order_identifier: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Marks a pending donation and payment order as cancelled by user."""
        order = db.execute(
            select(PaymentOrder).where(
                (PaymentOrder.id == order_identifier) | (PaymentOrder.provider_order_id == order_identifier)
            )
        ).scalar_one_or_none()

        if not order:
            raise PaymentOrderNotFoundException(f"Payment order '{order_identifier}' not found")

        donation = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
        if not donation:
            raise NotFoundException("Associated donation record not found")

        if donation.donor_id and user_id and donation.donor_id != user_id:
            user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
            if not user or user.platform_role != PlatformRole.PLATFORM_ADMIN:
                raise ForbiddenException("Access denied: You do not have permission to cancel this donation order.")

        if donation.status == DonationStatus.PAID:
            raise ValidationException("Cannot cancel a completed donation. Request a refund instead.")

        if donation.status in (DonationStatus.CANCELLED, DonationStatus.FAILED):
            return {"status": donation.status.value, "message": "Donation was already cancelled"}

        StateMachineValidator.validate_payment_order_transition(order.status, PaymentOrderStatus.VOIDED)
        order.status = PaymentOrderStatus.VOIDED

        StateMachineValidator.validate_donation_transition(donation.status, DonationStatus.CANCELLED)
        donation.status = DonationStatus.CANCELLED
        donation.cancelled_at = datetime.now(timezone.utc)

        db.commit()
        return {"status": "CANCELLED", "donation_id": donation.id, "order_id": order.id}

    # -------------------------------------------------------------------------
    # 4. Refund Processing (Full & Partial)
    # -------------------------------------------------------------------------

    def process_refund(
        self,
        db: Session,
        donation_id: str,
        amount_minor_units: int,
        reason: str,
        actor_id: str,
        is_admin: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a partial or full refund against a captured donation via the PayPal Payments v2 API.
        Enforces strict eligibility checks: amount > 0, cumulative refunds <= captured amount.
        Decrements net project and milestone funding totals upon completion.
        """
        donation = db.execute(select(Donation).where(Donation.id == donation_id)).scalar_one_or_none()
        if not donation:
            raise NotFoundException(f"Donation with ID '{donation_id}' not found")

        if not is_admin and donation.donor_id != actor_id:
            raise ForbiddenException("Access denied: You can only request refunds for your own donations.")

        if donation.status not in (DonationStatus.PAID, DonationStatus.PARTIALLY_REFUNDED):
            raise ValidationException(
                f"Donation is not in a refundable state (status: {donation.status.value}). Only PAID donations can be refunded."
            )

        # 1. Locate successful PaymentCapture
        order = db.execute(select(PaymentOrder).where(PaymentOrder.donation_id == donation.id)).scalar_one_or_none()
        if not order:
            raise NotFoundException("Associated payment order not found")

        capture = db.execute(select(PaymentCapture).where(PaymentCapture.order_id == order.id)).scalar_one_or_none()
        if not capture or capture.status not in (PaymentCaptureStatus.COMPLETED, PaymentCaptureStatus.PARTIALLY_REFUNDED):
            raise ValidationException("No settled payment capture found for this donation.")

        # 2. Calculate remaining refundable amount
        existing_refunds = db.execute(
            select(PaymentRefund).where(
                and_(PaymentRefund.capture_id == capture.id, PaymentRefund.status == PaymentRefundStatus.COMPLETED)
            )
        ).scalars().all()
        already_refunded = sum(r.amount_minor_units for r in existing_refunds)
        remaining_refundable = capture.amount_minor_units - already_refunded

        if amount_minor_units <= 0:
            raise InvalidPaymentAmountException("Refund amount must be greater than zero.")

        if amount_minor_units > remaining_refundable:
            raise PaymentRefundFailedException(
                f"Refund amount ({amount_minor_units} minor units) exceeds remaining refundable balance ({remaining_refundable} minor units)."
            )

        # 3. Call PayPal Refund API
        refund_money = Money(minor_units=amount_minor_units, currency=donation.currency)
        idempotency_key = str(uuid.uuid4())

        try:
            gateway_res = self.provider.refund_capture(
                capture_id=capture.provider_capture_id,
                amount=refund_money,
                reason=reason,
                idempotency_key=idempotency_key
            )
        except PaymentException as exc:
            logger.error("PayPal refund request rejected", exc_info=True)
            raise exc

        provider_refund_id = gateway_res.get("refund_id") or gateway_res.get("id")
        if not provider_refund_id:
            raise PaymentRefundFailedException(
                f"PayPal refund processed but no genuine provider refund ID was returned for capture {capture.provider_capture_id}"
            )

        now_utc = datetime.now(timezone.utc)
        gateway_status = (gateway_res.get("status") or "COMPLETED").upper()

        if gateway_status == "COMPLETED":
            refund_status = PaymentRefundStatus.COMPLETED
        elif gateway_status == "PENDING":
            refund_status = PaymentRefundStatus.PENDING
        else:
            raise PaymentRefundFailedException(f"PayPal refund rejected with gateway status: {gateway_status}")

        # 4. Persist PaymentRefund record
        refund_record = PaymentRefund(
            id=str(uuid.uuid4()),
            capture_id=capture.id,
            provider_refund_id=provider_refund_id,
            amount_minor_units=amount_minor_units,
            currency=donation.currency,
            reason=reason,
            status=refund_status,
            idempotency_key=idempotency_key,
            refunded_at=now_utc if refund_status == PaymentRefundStatus.COMPLETED else None
        )
        db.add(refund_record)

        # 5. Update Capture, Donation, and net funding totals only when refund is COMPLETED
        if refund_status == PaymentRefundStatus.COMPLETED:
            new_total_refunded = already_refunded + amount_minor_units
            if new_total_refunded >= capture.amount_minor_units:
                capture.status = PaymentCaptureStatus.REFUNDED
                StateMachineValidator.validate_donation_transition(donation.status, DonationStatus.REFUNDED)
                donation.status = DonationStatus.REFUNDED
            else:
                capture.status = PaymentCaptureStatus.PARTIALLY_REFUNDED
                StateMachineValidator.validate_donation_transition(donation.status, DonationStatus.PARTIALLY_REFUNDED)
                donation.status = DonationStatus.PARTIALLY_REFUNDED

            # 6. Adjust project and milestone net funding totals
            project = db.execute(select(Project).where(Project.id == donation.project_id)).scalar_one_or_none()
            milestone = db.execute(select(Milestone).where(Milestone.id == donation.milestone_id)).scalar_one_or_none()

            if milestone:
                milestone.raised_amount_minor_units = max(0, (milestone.raised_amount_minor_units or 0) - amount_minor_units)
            if project:
                project.raised_amount_minor_units = max(0, (project.raised_amount_minor_units or 0) - amount_minor_units)

        # 7. Audit trail
        audit = AuditEvent(
            id=str(uuid.uuid4()),
            actor_id=actor_id,
            action="DONATION_REFUND_COMPLETED",
            entity_type="PaymentRefund",
            entity_id=refund_record.id,
            ip_address="internal",
            details_json=json.dumps({
                "donation_id": donation.id,
                "capture_id": capture.id,
                "refund_id": refund_record.id,
                "amount_minor_units": amount_minor_units,
                "currency": donation.currency,
                "reason": reason
            })
        )
        db.add(audit)
        db.commit()
        db.refresh(refund_record)

        logger.info(
            f"Successfully processed refund of {amount_minor_units} for Donation {donation.id}",
            extra={"refund_id": refund_record.id, "provider_refund_id": provider_refund_id}
        )

        return {
            "refund_id": refund_record.id,
            "donation_id": donation.id,
            "provider_refund_id": refund_record.provider_refund_id,
            "amount_minor_units": refund_record.amount_minor_units,
            "currency": refund_record.currency,
            "status": refund_record.status.value,
            "donation_status": donation.status.value,
            "remaining_refundable_minor_units": capture.amount_minor_units - new_total_refunded
        }

    # -------------------------------------------------------------------------
    # 5. PayPal Webhook Processing & Deduplication
    # -------------------------------------------------------------------------

    def process_paypal_webhook(
        self,
        db: Session,
        headers: Dict[str, str],
        raw_body: bytes
    ) -> Dict[str, Any]:
        """
        Receives, cryptographically verifies, and safely processes incoming PayPal webhooks.
        Implements strict event deduplication to ensure idempotency.
        """
        try:
            body_json = json.loads(raw_body.decode("utf-8"))
        except Exception:
            raise ValidationException("Invalid webhook payload: not valid JSON")

        event_id = body_json.get("id")
        event_type = body_json.get("event_type")
        resource = body_json.get("resource", {})

        if not event_id or not event_type:
            raise ValidationException("Webhook payload missing 'id' or 'event_type'")

        # 1. Deduplication Check
        existing_event = db.execute(select(WebhookEvent).where(WebhookEvent.event_id == event_id)).scalar_one_or_none()
        if existing_event and existing_event.processing_status == WebhookProcessingStatus.PROCESSED:
            logger.info(f"Webhook event {event_id} already processed. Acknowledging duplicate.")
            return {"status": "DUPLICATE", "event_id": event_id, "message": "Event already processed"}

        # 2. Cryptographic Signature Verification
        is_signature_valid = self.provider.verify_webhook_signature(headers, raw_body)
        if not is_signature_valid:
            logger.warning(f"Rejected unverified webhook signature for event {event_id}")
            if not existing_event:
                failed_event = WebhookEvent(
                    id=str(uuid.uuid4()),
                    provider="PAYPAL",
                    event_id=event_id,
                    event_type=event_type,
                    resource_id=resource.get("id"),
                    resource_type=body_json.get("resource_type"),
                    signature_verification_status=WebhookSignatureStatus.FAILED,
                    processing_status=WebhookProcessingStatus.FAILED,
                    payload=json.dumps(body_json),
                    failure_reason="Cryptographic signature verification failed"
                )
                db.add(failed_event)
                db.commit()
            raise WebhookVerificationFailedException("PayPal webhook signature verification failed.")

        # 3. Create or update WebhookEvent record
        if not existing_event:
            event_record = WebhookEvent(
                id=str(uuid.uuid4()),
                provider="PAYPAL",
                event_id=event_id,
                event_type=event_type,
                resource_id=resource.get("id"),
                resource_type=body_json.get("resource_type"),
                signature_verification_status=WebhookSignatureStatus.VERIFIED,
                processing_status=WebhookProcessingStatus.PENDING,
                payload=json.dumps(body_json)
            )
            db.add(event_record)
            db.flush()
        else:
            event_record = existing_event

        # 4. Handle Event by Type
        try:
            if event_type == "CHECKOUT.ORDER.APPROVED":
                # Order approved by donor. Capture automatically if not yet captured.
                order_id = resource.get("id")
                if order_id:
                    order = db.execute(select(PaymentOrder).where(PaymentOrder.provider_order_id == order_id)).scalar_one_or_none()
                    if order and order.status != PaymentOrderStatus.COMPLETED:
                        logger.info(f"Auto-capturing approved order {order_id} via webhook")
                        self.capture_donation_payment(db, order_id)

            elif event_type == "PAYMENT.CAPTURE.COMPLETED":
                # Ensure local record is marked PAID
                capture_id = resource.get("id")
                custom_id = resource.get("custom_id")
                order = None
                if custom_id:
                    order = db.execute(select(PaymentOrder).where(PaymentOrder.donation_id == custom_id)).scalar_one_or_none()
                if not order and capture_id:
                    cap = db.execute(select(PaymentCapture).where(PaymentCapture.provider_capture_id == capture_id)).scalar_one_or_none()
                    if cap:
                        order = db.execute(select(PaymentOrder).where(PaymentOrder.id == cap.order_id)).scalar_one_or_none()

                if order and order.status != PaymentOrderStatus.COMPLETED:
                    self.capture_donation_payment(db, order.id)

            elif event_type in ("PAYMENT.CAPTURE.DENIED", "PAYMENT.CAPTURE.DECLINED"):
                capture_id = resource.get("id")
                cap = db.execute(select(PaymentCapture).where(PaymentCapture.provider_capture_id == capture_id)).scalar_one_or_none()
                if cap:
                    cap.status = PaymentCaptureStatus.DECLINED
                    order = db.execute(select(PaymentOrder).where(PaymentOrder.id == cap.order_id)).scalar_one_or_none()
                    if order:
                        order.status = PaymentOrderStatus.VOIDED
                        don = db.execute(select(Donation).where(Donation.id == order.donation_id)).scalar_one_or_none()
                        if don and don.status != DonationStatus.PAID:
                            don.status = DonationStatus.FAILED

            elif event_type in ("PAYMENT.CAPTURE.REFUNDED", "PAYMENT.CAPTURE.REVERSED"):
                # A refund was issued from PayPal dashboard
                logger.info(f"Received external refund webhook {event_id} for capture {resource.get('id')}")

            event_record.processing_status = WebhookProcessingStatus.PROCESSED
            event_record.processed_at = datetime.now(timezone.utc)
            db.commit()

        except Exception as exc:
            db.rollback()
            event_record.processing_status = WebhookProcessingStatus.FAILED
            event_record.failure_reason = str(exc)
            event_record.retry_count = (event_record.retry_count or 0) + 1
            db.commit()
            logger.error(f"Error processing webhook event {event_id}", exc_info=True)
            raise exc

        return {"status": "SUCCESS", "event_id": event_id, "event_type": event_type}

    # -------------------------------------------------------------------------
    # 6. Queries & Aggregations
    # -------------------------------------------------------------------------

    def get_project_funding_summary(self, db: Session, project_id: str) -> Dict[str, Any]:
        """Calculates accurate net funding aggregates for a project and its individual milestones."""
        project = db.execute(select(Project).where(Project.id == project_id)).scalar_one_or_none()
        if not project:
            raise NotFoundException(f"Project '{project_id}' not found")

        # Sum of successful captures for this project
        captures_sum = db.execute(
            select(func.coalesce(func.sum(PaymentCapture.amount_minor_units), 0))
            .select_from(PaymentCapture)
            .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
            .join(Donation, PaymentOrder.donation_id == Donation.id)
            .where(
                and_(
                    Donation.project_id == project_id,
                    PaymentCapture.status.in_([
                        PaymentCaptureStatus.COMPLETED,
                        PaymentCaptureStatus.PARTIALLY_REFUNDED,
                        PaymentCaptureStatus.REFUNDED
                    ])
                )
            )
        ).scalar() or 0

        # Sum of completed refunds for this project
        refunds_sum = db.execute(
            select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
            .select_from(PaymentRefund)
            .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
            .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
            .join(Donation, PaymentOrder.donation_id == Donation.id)
            .where(
                and_(
                    Donation.project_id == project_id,
                    PaymentRefund.status == PaymentRefundStatus.COMPLETED
                )
            )
        ).scalar() or 0

        net_funding = captures_sum - refunds_sum

        # Aggregate per milestone
        milestones = db.execute(select(Milestone).where(Milestone.project_id == project_id).order_by(Milestone.sequence_number)).scalars().all()
        milestone_data = []
        for m in milestones:
            m_captures = db.execute(
                select(func.coalesce(func.sum(PaymentCapture.amount_minor_units), 0))
                .select_from(PaymentCapture)
                .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
                .join(Donation, PaymentOrder.donation_id == Donation.id)
                .where(
                    and_(
                        Donation.milestone_id == m.id,
                        PaymentCapture.status.in_([
                            PaymentCaptureStatus.COMPLETED,
                            PaymentCaptureStatus.PARTIALLY_REFUNDED,
                            PaymentCaptureStatus.REFUNDED
                        ])
                    )
                )
            ).scalar() or 0

            m_refunds = db.execute(
                select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
                .select_from(PaymentRefund)
                .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
                .join(PaymentOrder, PaymentCapture.order_id == PaymentOrder.id)
                .join(Donation, PaymentOrder.donation_id == Donation.id)
                .where(
                    and_(
                        Donation.milestone_id == m.id,
                        PaymentRefund.status == PaymentRefundStatus.COMPLETED
                    )
                )
            ).scalar() or 0

            m_net = m_captures - m_refunds
            milestone_data.append({
                "milestone_id": m.id,
                "sequence_number": m.sequence_number,
                "title": m.title,
                "target_amount_minor_units": m.target_amount_minor_units,
                "gross_received_minor_units": m_captures,
                "refunded_minor_units": m_refunds,
                "net_received_minor_units": m_net,
                "percentage_funded": round((m_net / m.target_amount_minor_units * 100), 1) if m.target_amount_minor_units > 0 else 0.0
            })

        return {
            "project_id": project.id,
            "project_title": project.title,
            "currency": project.currency,
            "target_amount_minor_units": project.target_amount_minor_units,
            "gross_received_minor_units": captures_sum,
            "refunded_minor_units": refunds_sum,
            "net_received_minor_units": net_funding,
            "percentage_funded": round((net_funding / project.target_amount_minor_units * 100), 1) if project.target_amount_minor_units > 0 else 0.0,
            "milestones": milestone_data
        }

    def list_donor_donations(
        self,
        db: Session,
        donor_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Donation], int]:
        """Returns paginated donations made by an authenticated donor."""
        offset = (page - 1) * page_size
        query = select(Donation).where(Donation.donor_id == donor_id).order_by(Donation.created_at.desc())
        total = db.execute(select(func.count(Donation.id)).where(Donation.donor_id == donor_id)).scalar() or 0
        items = db.execute(query.offset(offset).limit(page_size)).scalars().all()
        return items, total

    def get_donation(self, db: Session, donation_id: str, user_id: Optional[str] = None) -> Donation:
        """Retrieves donation details with access control verification."""
        donation = db.execute(select(Donation).where(Donation.id == donation_id)).scalar_one_or_none()
        if not donation:
            raise NotFoundException(f"Donation '{donation_id}' not found")

        if donation.donor_id and user_id and donation.donor_id != user_id:
            user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
            if not user or user.platform_role != PlatformRole.PLATFORM_ADMIN:
                raise ForbiddenException("Access denied: You do not own this donation record.")

        return donation
