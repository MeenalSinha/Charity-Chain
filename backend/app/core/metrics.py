"""
metrics.py - Prometheus Metrics & Operational Telemetry
=======================================================
Defines standard Prometheus counters, gauges, and histograms for HTTP requests,
financial operations, AI forensic processing, rate limiting, and security events.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)

# 1. HTTP Server Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests handled by FastAPI",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# 2. Financial & Payment Metrics
donations_captured_total = Counter(
    "charitychain_donations_captured_total",
    "Total count of successfully captured donations",
    ["currency"],
)

donations_captured_minor_units = Counter(
    "charitychain_donations_captured_minor_units_total",
    "Total donation volume captured in integer minor units (e.g. cents)",
    ["currency"],
)

refunds_processed_total = Counter(
    "charitychain_refunds_processed_total",
    "Total count of processed donation refunds",
    ["currency"],
)

# 3. AI & Evidence Verification Metrics
ai_verification_duration_seconds = Histogram(
    "charitychain_ai_verification_duration_seconds",
    "Processing time for AI evidence forensic pipelines in seconds",
    ["provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 60.0],
)

ai_verification_tampering_detections_total = Counter(
    "charitychain_ai_verification_tampering_detections_total",
    "Total evidence items flagged for potential tampering (ELA or duplicate pHash)",
    ["tamper_type"],
)

# 4. Rate Limiting & Security Metrics
rate_limit_hits_total = Counter(
    "charitychain_rate_limit_hits_total",
    "Total requests throttled by sliding-window rate limiters",
    ["tier"],
)

security_audit_events_total = Counter(
    "charitychain_security_audit_events_total",
    "Count of recorded security audit events",
    ["severity"],
)

# 5. Outbox & Communications Metrics
outbox_events_processed_total = Counter(
    "charitychain_outbox_events_processed_total",
    "Total transactional outbox events processed",
    ["status"],
)

outbox_dead_letter_total = Counter(
    "charitychain_outbox_dead_letter_total",
    "Total outbox events transitioned to dead-letter queue",
    ["event_type"],
)


def get_latest_metrics() -> bytes:
    """Generates the latest Prometheus exposition metrics."""
    return generate_latest(REGISTRY)

