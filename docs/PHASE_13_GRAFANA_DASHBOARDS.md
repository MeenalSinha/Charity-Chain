# CharityChain — Phase 13 Grafana Dashboards & Alerting
**Document Version:** 1.0.0  
**Directory:** `docker/grafana/dashboards/`  
**Provisioning Provider:** `docker/grafana/provisioning/dashboards/dashboards.yml`  

---

## 1. Overview

CharityChain includes five production-ready, auto-provisioned Grafana dashboards providing real-time visibility for platform operators, security engineers, and financial auditors.

Dashboards are stored as declarative JSON specifications, version-controlled, and mounted automatically when Grafana starts.

---

## 2. Dashboard Inventory

### 2.1 Dashboard 1: Platform Health & HTTP Telemetry (`cc-app-health`)
- **HTTP Request Throughput**: Rate of incoming requests grouped by HTTP method (`rate(http_requests_total[1m])`).
- **HTTP Error Rates**: Real-time rate of client errors (4xx) and internal server errors (5xx).
- **Latency Percentiles**: P50 (median), P95, and P99 latency percentiles calculated via `histogram_quantile`.
- **Total Invocations**: Cumulative count of processed requests.

### 2.2 Dashboard 2: Financial Operations & Payments (`cc-financial-ops`)
- **Donation Capture Rate**: Successful payments per second segmented by ISO currency.
- **Gross Captured Volume**: Total USD volume captured in integer cents.
- **Refund Throughput**: Processed refunds over 10-minute sliding windows.
- **Refund Ratio Gauge**: Ratio of refunded payments relative to gross captures, alerting if $> 2\%$.

### 2.3 Dashboard 3: AI Forensic Pipeline Telemetry (`cc-ai-evidence`)
- **Forensic Pipeline Latency**: P95 latency of YOLO object detection, ELA, and pHash evaluations.
- **Tampering Flags Rate**: Frequency of evidence submissions flagged for ELA deviations or image duplicates.
- **Flagged Anomalies Gauge**: Cumulative count of suspicious submissions requiring reviewer escalation.

### 2.4 Dashboard 4: Security & Abuse Telemetry (`cc-security-telemetry`)
- **Rate Limit Drops**: Rate of requests rejected with HTTP 429 across `auth`, `financial`, and `evidence` tiers.
- **Audit Event Ingestion**: Volume of security audit events ingested into the SHA-256 hash chain by severity.
- **Critical Security Alerts**: Visual indicator of high-severity events (`AUTH_LOCKOUT`, `AUTH_TOKEN_REUSE`, `ROLE_CHANGED`).

### 2.5 Dashboard 5: Transactional Outbox & Communications (`cc-communications`)
- **Outbox Event Throughput**: Velocity of transactional emails and webhook dispatches.
- **Dead-Letter Accumulation**: Events failing after maximum retries.
- **Total Dead-Letter Counter**: Monitored count triggering immediate operational alarms when non-zero.

---

## 3. Recommended Production Alerting Rules

| Alert Rule Name | Condition | Severity | Action |
|---|---|:---:|---|
| `HighHttp5xxRate` | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05` | `CRITICAL` | Page on-call engineering; inspect Sentry |
| `P95LatencyDegradation` | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 2.0` | `WARNING` | Check DB pool exhaustion or CPU spikes |
| `BruteForceAuthSpike` | `rate(charitychain_rate_limit_hits_total{tier="auth"}[1m]) > 10` | `WARNING` | Check IP range for credential stuffing attack |
| `DeadLetterEventsDetected` | `increase(charitychain_outbox_dead_letter_total[5m]) > 0` | `CRITICAL` | Inspect failing webhook subscriptions or SMTP sink |
| `AuditChainTamperAlert` | Security audit verification endpoint returns `is_valid: false` | `P0 CATASTROPHIC` | Freeze admin access; initiate Incident Response Plan |
