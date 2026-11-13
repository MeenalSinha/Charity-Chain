"""
test_analytics_exports.py - CSV Formula Injection Sanitization Tests
====================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import pytest
from backend.app.services.analytics.analytics_export_service import AnalyticsExportService


def test_csv_formula_injection_prevention():
    """Verifies that malicious cell inputs starting with =, +, -, @, tab, or CR are neutralized."""
    malicious_inputs = [
        "=cmd|'/C calc'!A0",
        "+1+1",
        "-5+2",
        "@SUM(1,2)",
        "\tmalicious_tab",
        "\rmalicious_cr"
    ]

    for payload in malicious_inputs:
        sanitized = AnalyticsExportService.sanitize_csv_cell(payload)
        assert sanitized.startswith("'"), f"Failed to sanitize: {payload}"
        assert sanitized == f"'{payload}"


def test_safe_inputs_remain_unmodified():
    """Verifies that benign strings and numbers are preserved without quote prefixing."""
    safe_inputs = [
        "Clean Water Initiative",
        "Project #1024",
        "45.50",
        "100",
        "San Francisco, CA"
    ]

    for text in safe_inputs:
        sanitized = AnalyticsExportService.sanitize_csv_cell(text)
        assert not sanitized.startswith("'")
        assert sanitized == text

