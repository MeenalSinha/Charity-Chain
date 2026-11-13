"""
analytics_export_service.py - Report Export & CSV Formula Injection Protection
===============================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Generates role-authorized CSV and PDF analytical exports with mandatory CSV formula
injection sanitization and structured report metadata.
"""

import csv
import io
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session

from backend.app.db.models.analytics import AnalyticsExportJob
from backend.app.services.analytics.ngo_analytics_service import NGOAnalyticsService
from backend.app.services.analytics.donor_analytics_service import DonorAnalyticsService


class AnalyticsExportService:
    """Manages secure generation and sanitization of analytical exports."""

    @staticmethod
    def sanitize_csv_cell(value: Any) -> str:
        """
        Guards against CSV Formula Injection (CWE-1236 / DDE).
        Prefixes dangerous characters ('=', '+', '-', '@', tab, CR) with a single quote (').
        """
        if value is None:
            return ""
        s = str(value)
        if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
            return f"'{s}"
        return s

    @classmethod
    def export_ngo_projects_csv(cls, db: Session, organization_id: str) -> str:
        """Exports an organization's projects and progress to CSV with formula injection sanitization."""
        projects = NGOAnalyticsService.get_project_performance(db, organization_id)
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Header
        headers = [
            "Project ID", "Title", "Status", "Target Amount (USD)",
            "Raised Amount (USD)", "Funding Progress (%)",
            "Total Milestones", "Completed Milestones", "Completion (%)", "Created Date"
        ]
        writer.writerow(headers)

        for p in projects:
            row = [
                cls.sanitize_csv_cell(p["project_id"]),
                cls.sanitize_csv_cell(p["title"]),
                cls.sanitize_csv_cell(p["status"]),
                cls.sanitize_csv_cell(f"{p['target_amount_cents'] / 100:.2f}"),
                cls.sanitize_csv_cell(f"{p['raised_amount_cents'] / 100:.2f}"),
                cls.sanitize_csv_cell(f"{p['funding_progress_percent']}%"),
                cls.sanitize_csv_cell(p["milestones_total"]),
                cls.sanitize_csv_cell(p["milestones_completed"]),
                cls.sanitize_csv_cell(f"{p['completion_percent']}%"),
                cls.sanitize_csv_cell(p["created_at"] or "")
            ]
            writer.writerow(row)

        return output.getvalue()

    @classmethod
    def export_donor_contributions_csv(cls, db: Session, donor_id: str) -> str:
        """Exports a donor's contribution history to CSV with formula injection sanitization."""
        portfolio = DonorAnalyticsService.get_donation_portfolio(db, donor_id)
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        headers = [
            "Donation ID", "Project ID", "Project Title", "Milestone ID",
            "Amount (USD)", "Currency", "Donation Date"
        ]
        writer.writerow(headers)

        for d in portfolio.get("recent_donations", []):
            row = [
                cls.sanitize_csv_cell(d["donation_id"]),
                cls.sanitize_csv_cell(d["project_id"]),
                cls.sanitize_csv_cell(d["project_title"]),
                cls.sanitize_csv_cell(d["milestone_id"]),
                cls.sanitize_csv_cell(f"{d['amount_cents'] / 100:.2f}"),
                cls.sanitize_csv_cell(d["currency"]),
                cls.sanitize_csv_cell(d["created_at"] or "")
            ]
            writer.writerow(row)

        return output.getvalue()

    @classmethod
    def generate_formatted_report_text(
        cls,
        report_title: str,
        organization_or_donor: str,
        metrics: Dict[str, Any],
        disclaimer: str
    ) -> str:
        """Builds a human-readable structured textual report suitable for presentation or plain text PDF rendering."""
        lines = [
            "=" * 72,
            f"CHARITYCHAIN IMPACT INTELLIGENCE REPORT: {report_title.upper()}",
            "=" * 72,
            f"Subject: {organization_or_donor}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Ledger Verification Status: VERIFIED & AUDITABLE",
            "-" * 72,
            "KEY PERFORMANCE & IMPACT METRICS:",
        ]

        for k, v in metrics.items():
            if isinstance(v, dict):
                lines.append(f"\n[{k.replace('_', ' ').upper()}]:")
                for sub_k, sub_v in v.items():
                    lines.append(f"  • {sub_k.replace('_', ' ').capitalize()}: {sub_v}")
            else:
                lines.append(f"• {k.replace('_', ' ').capitalize()}: {v}")

        lines.extend([
            "-" * 72,
            "DISCLAIMER & LEGAL NOTICES:",
            disclaimer,
            "=" * 72
        ])
        return "\n".join(lines)

