# CharityChain — Phase 13 Prometheus Metrics & Observability
**Document Version:** 1.0.0  
**Implementation File:** `backend/app/core/metrics.py`  
**Endpoint:** `GET /metrics`  

---

## 1. Overview & Objectives

Prometheus instrumentation provides low-overhead, real-time telemetry across CharityChain's core operational vectors:
1. HTTP API request volume, error rates, and latency distributions.
2. Financial donation captures, volume aggregates, and refund frequency.
3. AI evidence forensic verification latency and tamper anomaly detection.
4. Security sliding-window rate limit throttling and audit event ingestion.
5. Transactional outbox event dispatch and dead-letter queue depth.

---

## 2. Canonical Metrics Catalog

### 2.1 HTTP & Server Metrics
- `http_requests_total`: `Counter`
  - *Labels*: `method` (GET, POST, etc.), `endpoint` (/api/v1/projects, etc.), `status` (200, 400, 429, 500)
  - *Description*: Count of HTTP requests processed by FastAPI.
- `http_request_duration_seconds`: `Histogram`
  - *Labels*: `method`, `endpoint`
  - *Buckets*: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]`
  - *Description*: Wall-clock execution time for HTTP requests.

### 2.2 Financial & Payment Metrics
- `charitychain_donations_captured_total`: `Counter`
  - *Labels*: `currency` (USD, EUR, GBP, CAD, AUD)
  - *Description*: Cumulative count of successfully captured donations.
- `charitychain_donations_captured_minor_units_total`: `Counter`
  - *Labels*: `currency`
  - *Description*: Cumulative monetary value in integer minor units (e.g. cents).
- `charitychain_refunds_processed_total`: `Counter`
  - *Labels*: `currency`
  - *Description*: Cumulative count of processed refunds.

### 2.3 AI & Evidence Forensic Metrics
- `charitychain_ai_verification_duration_seconds`: `Histogram`
  - *Labels*: `provider` (`yolo_forensics`, etc.)
  - *Buckets*: `[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 60.0]`
  - *Description*: Time elapsed during image analysis, ELA, pHash, and object detection.
- `charitychain_ai_verification_tampering_detections_total`: `Counter`
  - *Labels*: `tamper_type` (`ela_anomaly`, `duplicate_phash`)
  - *Description*: Total evidence media files flagged for potential tampering.

### 2.4 Abuse Prevention & Security Metrics
- `charitychain_rate_limit_hits_total`: `Counter`
  - *Labels*: `tier` (`auth`, `financial`, `evidence`, `default`)
  - *Description*: Requests throttled with HTTP 429 Too Many Requests.
- `charitychain_security_audit_events_total`: `Counter`
  - *Labels*: `severity` (`INFO`, `WARNING`, `ERROR`, `CRITICAL`)
  - *Description*: Ingested events in the tamper-evident security audit chain.

### 2.5 Outbox & Transactional Communications
- `charitychain_outbox_events_processed_total`: `Counter`
  - *Labels*: `status` (`SENT`, `FAILED`, `DEAD_LETTER`)
  - *Description*: Cumulative count of processed domain event notifications.
- `charitychain_outbox_dead_letter_total`: `Counter`
  - *Labels*: `event_type`
  - *Description*: Events permanently transitioned to dead-letter storage after exhausting retries.

---

## 3. Scraping Configuration (`docker/prometheus/prometheus.yml`)

Prometheus scrapes the backend service every 15 seconds:
```yaml
scrape_configs:
  - job_name: "charitychain-backend"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["backend:8000"]
```
