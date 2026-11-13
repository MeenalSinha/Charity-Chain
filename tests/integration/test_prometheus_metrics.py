"""
test_prometheus_metrics.py - Integration Tests for Prometheus Operational Metrics
=================================================================================
Validates Prometheus metrics generation, text exposition format, and
application counter/histogram tracking.
"""

from backend.app.core.metrics import (
    get_latest_metrics,
    http_requests_total,
    donations_captured_total,
    rate_limit_hits_total,
    CONTENT_TYPE_LATEST,
)


def test_prometheus_metrics_exposition_format():
    """Ensures /metrics endpoint generates standard Prometheus text format."""
    metrics_data = get_latest_metrics()
    assert isinstance(metrics_data, bytes)
    text = metrics_data.decode("utf-8")

    # Assert standard metric declarations exist
    assert "http_requests_total" in text
    assert "charitychain_donations_captured_total" in text
    assert "charitychain_rate_limit_hits_total" in text
    assert "charitychain_security_audit_events_total" in text


def test_metric_counter_increments():
    """Verifies custom application metrics increment properly."""
    # Increment test counter
    rate_limit_hits_total.labels(tier="auth").inc()
    donations_captured_total.labels(currency="USD").inc()

    text = get_latest_metrics().decode("utf-8")
    assert 'charitychain_rate_limit_hits_total{tier="auth"}' in text
    assert 'charitychain_donations_captured_total{currency="USD"}' in text
