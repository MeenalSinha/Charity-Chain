"""
email_template_service.py - Responsive Transactional Email Template Engine
==========================================================================
Renders branded, responsive HTML and plain-text emails for CharityChain notifications.
Protects against HTML injection via strict variable escaping and enforces consistent
visual branding, action buttons, and regulatory footers.
"""

import html
from typing import Dict, Any, Tuple
from backend.app.core.config import settings


class EmailTemplateService:
    """Service rendering branded, versioned transactional email templates."""

    @staticmethod
    def _render_base_layout(title: str, content_html: str, action_url: str = None, action_text: str = None) -> str:
        """Encloses email body in CharityChain's responsive HTML master layout."""
        button_html = ""
        if action_url and action_text:
            safe_url = html.escape(action_url)
            safe_text = html.escape(action_text)
            button_html = f"""
            <div style="text-align: center; margin: 32px 0;">
                <a href="{safe_url}" style="background-color: #059669; color: #ffffff; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 15px; display: inline-block;">
                    {safe_text}
                </a>
            </div>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1f2937;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f3f4f6; padding: 40px 10px;">
        <tr>
            <td align="center">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); max-width: 600px; width: 100%;">
                    <!-- Header -->
                    <tr>
                        <td style="background-color: #064e3b; padding: 24px 32px; text-align: left;">
                            <h1 style="color: #ffffff; margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px;">
                                🌍 CharityChain
                            </h1>
                            <p style="color: #a7f3d0; margin: 4px 0 0 0; font-size: 13px;">
                                Give with Evidence. Release with Confidence.
                            </p>
                        </td>
                    </tr>
                    <!-- Content -->
                    <tr>
                        <td style="padding: 36px 32px 24px 32px; line-height: 1.6; font-size: 15px;">
                            <h2 style="color: #111827; font-size: 19px; margin-top: 0; margin-bottom: 20px;">
                                {html.escape(title)}
                            </h2>
                            {content_html}
                            {button_html}
                        </td>
                    </tr>
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f9fafb; padding: 20px 32px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280; text-align: center; line-height: 1.5;">
                            <p style="margin: 0 0 8px 0;">
                                This is an automated transactional security communication from CharityChain.
                            </p>
                            <p style="margin: 0;">
                                © 2026 CharityChain Platform. All cryptographic attestations and impact events are logged to the immutable impact ledger.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @classmethod
    def render(cls, template_key: str, context: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Renders an email template by key.
        Returns: (subject, html_body, plain_text_body)
        """
        template_alias = {
            "payment.captured": "donation.captured",
            "payment.refunded": "donation.refunded",
        }.get(template_key, template_key)
        method_name = f"_render_{template_alias.replace('.', '_')}"
        renderer = getattr(cls, method_name, None)
        if not renderer:
            # Fallback to generic template
            return cls._render_generic(template_key, context)
        return renderer(context)

    @classmethod
    def _render_generic(cls, template_key: str, context: Dict[str, Any]) -> Tuple[str, str, str]:
        title = context.get("title", f"CharityChain Notification: {template_key}")
        message = context.get("message", "An event has occurred on your CharityChain account.")
        action_url = context.get("action_url")
        action_text = context.get("action_text", "View in CharityChain")

        content_html = f"<p>{html.escape(message)}</p>"
        html_body = cls._render_base_layout(title, content_html, action_url, action_text)
        text_body = f"{title}\n\n{message}\n\n" + (f"Link: {action_url}\n\n" if action_url else "")
        return title, html_body, text_body

    @classmethod
    def _render_donation_captured(cls, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
        donor_name = ctx.get("donor_name", "Valued Supporter")
        amount = ctx.get("formatted_amount")
        if not amount and "amount_minor_units" in ctx:
            curr = ctx.get("currency", "USD")
            val = ctx["amount_minor_units"] / 100.0
            amount = f"${val:,.2f}" if curr == "USD" else f"{val:,.2f} {curr}"
        if not amount:
            amount = "$0.00"
        project_title = ctx.get("project_title", "CharityChain Project")
        milestone_title = ctx.get("milestone_title", "Milestone Delivery")
        order_id = ctx.get("order_id", "N/A")
        action_url = ctx.get("action_url", f"{settings.ISSUER_PUBLIC_URL}/donor/donations")

        subject = f"Donation Confirmed: {amount} to {project_title}"
        content_html = f"""
        <p>Dear {html.escape(donor_name)},</p>
        <p>Thank you for your generous contribution. Your payment of <strong>{html.escape(amount)}</strong> has been verified and securely recorded.</p>
        <div style="background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; margin: 20px 0; border-radius: 4px;">
            <p style="margin: 0 0 6px 0;"><strong>Project:</strong> {html.escape(project_title)}</p>
            <p style="margin: 0 0 6px 0;"><strong>Target Milestone:</strong> {html.escape(milestone_title)}</p>
            <p style="margin: 0;"><strong>Transaction Reference:</strong> {html.escape(order_id)}</p>
        </div>
        <p>In accordance with CharityChain's verification charter, your funds are safely committed and will only be disbursed to the NGO once multi-indicator AI verification and dual-key human review confirm evidence of physical impact.</p>
        """
        html_body = cls._render_base_layout("Donation Confirmed", content_html, action_url, "View My Contribution")
        text_body = f"Dear {donor_name},\n\nYour contribution of {amount} to '{project_title}' has been confirmed (Ref: {order_id}).\n\nTrack progress: {action_url}"
        return subject, html_body, text_body

    @classmethod
    def _render_certificate_issued(cls, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
        recipient = ctx.get("recipient_name", "Supporter")
        cert_number = ctx.get("certificate_number", "CC-CERT-000000")
        project_title = ctx.get("project_title", "Verified Project")
        impact_summary = ctx.get("impact_summary", "Milestone successfully completed and verified.")
        verification_url = ctx.get("verification_url", f"{settings.ISSUER_PUBLIC_URL}/public/certificates")
        download_url = ctx.get("download_url")

        subject = f"Your Proof-of-Impact Certificate is Ready ({cert_number})"
        content_html = f"""
        <p>Dear {html.escape(recipient)},</p>
        <p>We are delighted to present your official <strong>Proof-of-Impact Certificate</strong>. The project you supported has successfully completed its milestone with verified evidence!</p>
        <div style="background-color: #ecfdf5; border: 1px solid #10b981; padding: 20px; border-radius: 6px; margin: 24px 0; text-align: center;">
            <p style="font-size: 13px; color: #047857; margin: 0 0 4px 0; text-transform: uppercase; font-weight: bold;">Verified Certificate</p>
            <h3 style="margin: 0 0 8px 0; color: #065f46; font-size: 20px;">{html.escape(cert_number)}</h3>
            <p style="margin: 0; color: #374151;"><strong>Project:</strong> {html.escape(project_title)}</p>
            <p style="margin: 8px 0 0 0; color: #4b5563; font-size: 14px;"><em>"{html.escape(impact_summary)}"</em></p>
        </div>
        <p>This certificate is cryptographically signed using Ed25519 asymmetric cryptography and permanently logged on CharityChain's impact ledger. Anyone can scan your certificate's QR code to independently verify its authenticity without passwords.</p>
        """
        html_body = cls._render_base_layout("Proof-of-Impact Certificate Issued", content_html, verification_url, "Verify Digital Certificate")
        text_body = f"Dear {recipient},\n\nYour Proof-of-Impact Certificate ({cert_number}) for '{project_title}' has been issued.\n\nImpact: {impact_summary}\n\nVerify online: {verification_url}"
        return subject, html_body, text_body

    @classmethod
    def _render_certificate_revoked(cls, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
        cert_number = ctx.get("certificate_number", "CC-CERT-000000")
        reason = ctx.get("reason", "Milestone review status changed or full refund issued.")
        verification_url = ctx.get("verification_url", f"{settings.ISSUER_PUBLIC_URL}/public/certificates")

        subject = f"Notice: Certificate {cert_number} Status Changed"
        content_html = f"""
        <p>Hello,</p>
        <p>This is an automated notice that Proof-of-Impact Certificate <strong>{html.escape(cert_number)}</strong> has been updated to <strong>REVOKED</strong> on the CharityChain public impact ledger.</p>
        <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 16px; margin: 20px 0; border-radius: 4px;">
            <p style="margin: 0 0 6px 0;"><strong>Reason:</strong> {html.escape(reason)}</p>
        </div>
        <p>The public verification registry has been updated accordingly. If you have questions regarding this change, please consult the project governance dashboard.</p>
        """
        html_body = cls._render_base_layout("Certificate Status Update", content_html, verification_url, "View Status on Ledger")
        text_body = f"Notice: Certificate {cert_number} has been revoked.\n\nReason: {reason}\n\nCheck ledger: {verification_url}"
        return subject, html_body, text_body

    @classmethod
    def _render_review_assigned(cls, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
        reviewer_name = ctx.get("reviewer_name", "Reviewer")
        milestone_title = ctx.get("milestone_title", "Milestone Evidence")
        project_title = ctx.get("project_title", "Project")
        deadline = ctx.get("due_date", "72 hours")
        action_url = ctx.get("action_url", f"{settings.ISSUER_PUBLIC_URL}/review/queue")

        subject = f"New Review Assignment: {milestone_title}"
        content_html = f"""
        <p>Dear {html.escape(reviewer_name)},</p>
        <p>You have been assigned to conduct an independent verification review for <strong>{html.escape(milestone_title)}</strong> under project <em>{html.escape(project_title)}</em>.</p>
        <p>Multi-indicator AI forensic analysis is complete. Please inspect the photographs, EXIF metadata, and forensic indicators to submit your independent approval, revision request, or rejection.</p>
        <p><strong>Review Window:</strong> Please finalize your decision within {html.escape(deadline)}.</p>
        """
        html_body = cls._render_base_layout("New Review Assignment", content_html, action_url, "Open Reviewer Workspace")
        text_body = f"Dear {reviewer_name},\n\nYou have a new review assignment for '{milestone_title}' in '{project_title}'.\n\nPlease submit decision within {deadline}.\n\nReview link: {action_url}"
        return subject, html_body, text_body

    @classmethod
    def _render_disbursement_completed(cls, ctx: Dict[str, Any]) -> Tuple[str, str, str]:
        org_name = ctx.get("org_name", "Partner NGO")
        amount = ctx.get("formatted_amount", "$0.00")
        milestone_title = ctx.get("milestone_title", "Milestone Delivery")
        batch_id = ctx.get("batch_id", "N/A")
        action_url = ctx.get("action_url", f"{settings.ISSUER_PUBLIC_URL}/ngo/disbursements")

        subject = f"Disbursement Transferred: {amount} for {milestone_title}"
        content_html = f"""
        <p>Dear {html.escape(org_name)},</p>
        <p>We are pleased to inform you that the milestone payout of <strong>{html.escape(amount)}</strong> for <strong>{html.escape(milestone_title)}</strong> has been successfully processed and transferred to your registered payout account.</p>
        <div style="background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; margin: 20px 0; border-radius: 4px;">
            <p style="margin: 0 0 6px 0;"><strong>Amount:</strong> {html.escape(amount)}</p>
            <p style="margin: 0;"><strong>Payout Batch Reference:</strong> {html.escape(batch_id)}</p>
        </div>
        <p>All disbursements are automatically balanced on CharityChain's double-entry financial ledger.</p>
        """
        html_body = cls._render_base_layout("Disbursement Completed", content_html, action_url, "View Financial Summary")
        text_body = f"Dear {org_name},\n\nA disbursement of {amount} for '{milestone_title}' has been successfully processed (Ref: {batch_id}).\n\nDetails: {action_url}"
        return subject, html_body, text_body

    @classmethod
    def render_donation_receipt(
        cls,
        donor_name: str,
        project_title: str,
        amount_formatted: str,
        donation_id: str,
        receipt_url: str = None
    ) -> Dict[str, str]:
        """Convenience method rendering donation confirmation email."""
        subject, html_body, text_body = cls.render("donation.captured", {
            "donor_name": donor_name,
            "project_title": project_title,
            "formatted_amount": amount_formatted,
            "order_id": donation_id,
            "action_url": receipt_url
        })
        return {"subject": subject, "html_body": html_body, "text_body": text_body}

    @classmethod
    def render_certificate_issued(
        cls,
        recipient_name: str,
        certificate_number: str,
        project_title: str,
        verification_url: str,
        impact_summary: str = None
    ) -> Dict[str, str]:
        """Convenience method rendering certificate issuance email."""
        subject, html_body, text_body = cls.render("certificate.issued", {
            "recipient_name": recipient_name,
            "certificate_number": certificate_number,
            "project_title": project_title,
            "verification_url": verification_url,
            "impact_summary": impact_summary or "Milestone completed with verified impact."
        })
        return {"subject": subject, "html_body": html_body, "text_body": text_body}

    @classmethod
    def render_disbursement_completed(
        cls,
        organization_name: str,
        milestone_title: str,
        amount_formatted: str,
        payout_batch_id: str
    ) -> Dict[str, str]:
        """Convenience method rendering NGO disbursement confirmation email."""
        subject, html_body, text_body = cls.render("disbursement.completed", {
            "org_name": organization_name,
            "milestone_title": milestone_title,
            "formatted_amount": amount_formatted,
            "batch_id": payout_batch_id
        })
        return {"subject": subject, "html_body": html_body, "text_body": text_body}
