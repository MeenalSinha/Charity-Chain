"""
certificate_rendering_service.py - Professional PDF & QR Code Rendering
========================================================================
Generates publication-quality downloadable PDF certificates and high-density QR codes
linking directly to CharityChain's public verification endpoint.
Calculates SHA-256 PDF digests for tamper-evident storage and non-repudiation.
"""

import io
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import qrcode
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from backend.app.core.config import settings
from backend.app.integrations.storage import get_storage_provider


class CertificateRenderingService:
    @staticmethod
    def generate_qr_code_png(verification_url: str) -> bytes:
        """Generates a high-density PNG QR code byte stream pointing to the public verification URL."""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=2,
        )
        qr.add_data(verification_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#1E293B", back_color="white")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def render_certificate_pdf(
        cls,
        claims: Dict[str, Any],
        signature_b64: str,
        signing_key_id: str,
        verification_url: str
    ) -> Tuple[bytes, str]:
        """
        Renders a publication-grade landscape PDF certificate using ReportLab.
        Returns: (pdf_bytes, pdf_sha256_hash)
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "CertTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=30,
            textColor=colors.HexColor("#0F172A"),
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            "CertSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#64748B"),
            alignment=1
        )
        recipient_style = ParagraphStyle(
            "CertRecipient",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0284C7"),
            alignment=1
        )
        body_style = ParagraphStyle(
            "CertBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            alignment=1
        )
        small_meta_style = ParagraphStyle(
            "CertMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569")
        )
        disclaimer_style = ParagraphStyle(
            "CertDisclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#94A3B8"),
            alignment=1
        )

        elements = []

        # Header Branding
        elements.append(Paragraph("CHARITYCHAIN IMPACT VERIFICATION", subtitle_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph("PROOF-OF-IMPACT CERTIFICATE", title_style))
        elements.append(Paragraph("Give with Evidence. Release with Confidence.", subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

        # Recipient Designation
        donor_info = claims.get("donor", {})
        recipient_display = donor_info.get("display_name", "Anonymous Contributor")
        elements.append(Paragraph(f"Awarded with gratitude to", subtitle_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(recipient_display, recipient_style))
        elements.append(Spacer(1, 8))

        # Attestation Narrative
        proj = claims.get("project", {}).get("title", "Charitable Project")
        milestone = claims.get("milestone", {}).get("title", "Project Milestone")
        org = claims.get("organization", {}).get("name", "Verified Non-Profit")
        amt_cents = claims.get("contribution", {}).get("amount_cents", 0)
        currency = claims.get("contribution", {}).get("currency", "USD")
        amt_str = f"${amt_cents / 100:.2f} {currency}" if amt_cents else "Designated Contribution"

        narrative = (
            f"CharityChain records that a contribution of <b>{amt_str}</b> was allocated to "
            f"<b>{milestone}</b> under <b>{proj}</b>, managed by <b>{org}</b>. "
            f"The milestone subsequently satisfied all multi-indicator AI forensic evaluations "
            f"and passed final independent Dual-Key human review approval."
        )
        elements.append(Paragraph(narrative, body_style))
        elements.append(Spacer(1, 14))

        # QR Code & Technical Verification Metadata Table
        qr_bytes = cls.generate_qr_code_png(verification_url)
        qr_img = Image(io.BytesIO(qr_bytes), width=1.4 * inch, height=1.4 * inch)

        cert_num = claims.get("certificate_number", "N/A")
        issued_at = claims.get("issued_at", datetime.utcnow().isoformat())
        snapshot = claims.get("verification_summary", {}).get("snapshot_digest", "N/A")
        sig_preview = f"{signature_b64[:32]}...{signature_b64[-16:]}" if len(signature_b64) > 48 else signature_b64

        meta_text = (
            f"<b>Certificate Number:</b> {cert_num}<br/>"
            f"<b>Issued At:</b> {issued_at}<br/>"
            f"<b>Evidence Snapshot Digest:</b> {snapshot}<br/>"
            f"<b>Signature Algorithm:</b> Ed25519 (EdDSA) | <b>Key ID:</b> {signing_key_id}<br/>"
            f"<b>Digital Signature:</b> <font face='Courier'>{sig_preview}</font><br/>"
            f"<b>Scan QR Code to Verify Authenticity Online:</b> {verification_url}"
        )
        meta_paragraph = Paragraph(meta_text, small_meta_style)

        table_data = [[meta_paragraph, qr_img]]
        meta_table = Table(table_data, colWidths=[520, 150])
        meta_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#F8FAFC")),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (0, 0), 0.5, colors.HexColor("#E2E8F0")),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # Legal Disclaimer
        disclaimer = (
            "This document is a cryptographically verifiable digital impact attestation signed by CharityChain. "
            "It establishes documented compliance with the platform's multi-indicator evidence verification and human governance workflow. "
            "Internal milestone allocations represent accounting records and do not constitute a banking escrow trust. "
            "This certificate is not an official tax-deductible receipt unless explicitly certified under local regulatory jurisdiction."
        )
        elements.append(Paragraph(disclaimer, disclaimer_style))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        return pdf_bytes, pdf_hash

    @classmethod
    def save_pdf_artifact(
        cls,
        certificate_number: str,
        pdf_bytes: bytes
    ) -> str:
        """
        Saves the generated PDF certificate using the configured storage provider or local directory.
        Returns: storage_key
        """
        filename = f"{certificate_number.replace('-', '_')}.pdf"
        storage_key = f"certificates/{filename}"

        try:
            storage = get_storage_provider()
            storage.save(storage_key, pdf_bytes)
            return storage_key
        except Exception:
            # Fallback to local certificates directory
            local_dir = Path(getattr(settings, "CERTIFICATE_PDF_STORAGE_DIR", "data/certificates/pdf"))
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            with open(local_path, "wb") as f:
                f.write(pdf_bytes)
            return str(local_path)

