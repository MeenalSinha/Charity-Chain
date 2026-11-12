"""
test_email_templates.py - Unit Tests for Responsive HTML Email Rendering & XSS Safety
=====================================================================================
Verifies responsive HTML generation, plain text fallbacks, and HTML escaping of user inputs.
"""

import pytest
from backend.app.services.email_template_service import EmailTemplateService


def test_donation_receipt_rendering_and_escaping():
    """Verifies donation receipt template escapes malicious script tags."""
    xss_payload = "<script>alert('pwned')</script>"
    rendered = EmailTemplateService.render_donation_receipt(
        donor_name=xss_payload,
        project_title="Safe Water for All",
        amount_formatted="$50.00",
        donation_id="don-12345",
        receipt_url="https://charitychain.org/donations/receipt/don-12345"
    )

    assert "subject" in rendered
    assert "html_body" in rendered
    assert "text_body" in rendered

    # XSS injection must be escaped in HTML
    assert "<script>" not in rendered["html_body"]
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;" in rendered["html_body"]
    # Plain text should contain donor name
    assert xss_payload in rendered["text_body"]
    assert "$50.00" in rendered["html_body"]


def test_certificate_issuance_rendering():
    """Verifies certificate issued template structure."""
    rendered = EmailTemplateService.render_certificate_issued(
        recipient_name="Alice Smith",
        certificate_number="CC-2026-0001",
        project_title="School Solar Energy",
        verification_url="https://charitychain.org/verify/v-123",
        impact_summary="Generated 500 kWh of clean solar power."
    )

    assert "CC-2026-0001" in rendered["subject"]
    assert "Alice Smith" in rendered["html_body"]
    assert "https://charitychain.org/verify/v-123" in rendered["html_body"]
    assert "Verify Digital Certificate" in rendered["html_body"]


def test_disbursement_completed_rendering():
    """Verifies NGO payout notification email rendering."""
    rendered = EmailTemplateService.render_disbursement_completed(
        organization_name="Clean Earth NGO",
        milestone_title="Phase 1: Solar Panel Purchase",
        amount_formatted="$2,500.00",
        payout_batch_id="batch-999"
    )

    assert "$2,500.00" in rendered["subject"]
    assert "Clean Earth NGO" in rendered["html_body"]
    assert "batch-999" in rendered["html_body"]

