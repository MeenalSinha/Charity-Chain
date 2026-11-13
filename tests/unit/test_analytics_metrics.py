"""
test_analytics_metrics.py - Unit Tests for Metric Definitions Catalog
=====================================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
"""

import pytest
from backend.app.services.analytics.metric_definitions import (
    MetricDefinitionService, MetricPrivacy, METRIC_CATALOG
)


def test_metric_catalog_completeness():
    """Verifies that all core metrics exist in catalog with defined formulas and sources."""
    required_metrics = [
        "total_captured_donations",
        "net_donations_after_refunds",
        "confirmed_disbursements",
        "active_projects",
        "completed_projects",
        "total_milestones",
        "completed_milestones",
        "verified_milestones",
        "evidence_review_completion_rate",
        "donor_retention_rate",
        "total_certificates_issued"
    ]

    for m_id in required_metrics:
        metric = MetricDefinitionService.get_metric(m_id)
        assert metric is not None, f"Missing metric: {m_id}"
        assert metric.formula is not None
        assert metric.data_source is not None
        assert metric.unit in ("cents", "count", "percentage", "days")


def test_metric_privacy_filters():
    """Verifies that metrics are categorized by privacy tier."""
    public_metrics = MetricDefinitionService.list_metrics(MetricPrivacy.PUBLIC)
    admin_metrics = MetricDefinitionService.list_metrics(MetricPrivacy.ADMIN_ONLY)

    assert len(public_metrics) > 0
    assert len(admin_metrics) > 0

    # Ensure admin metrics are not labeled as public
    admin_ids = [m.metric_id for m in admin_metrics]
    for pm in public_metrics:
        assert pm.metric_id not in admin_ids


def test_metric_amount_formatting():
    """Verifies currency minor units formatting."""
    assert MetricDefinitionService.format_amount(100000, "USD") == "$1,000.00"
    assert MetricDefinitionService.format_amount(50, "USD") == "$0.50"
    assert MetricDefinitionService.format_amount(0, "USD") == "$0.00"
    assert MetricDefinitionService.format_amount(250000, "EUR") == "EUR 2,500.00"

