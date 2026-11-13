# CharityChain — Phase 13 Observability Architecture
**Document Version:** 1.0.0  
**Scope:** Telemetry, Tracing, Logging, Metrics, and Alerting  

---

## 1. Unified Observability Pillars

CharityChain's observability framework is built upon four synchronized pillars:

```
                          [ CHARITYCHAIN PLATFORM ]
                                      |
         +-----------------+----------+----------+-----------------+
         |                 |                     |                 |
         v                 v                     v                 v
   [ 1. METRICS ]    [ 2. LOGGING ]        [ 3. TRACING ]    [ 4. AUDIT TRAIL ]
   Prometheus        Structured JSON       Sentry APM &      SHA-256 Hash
   /metrics          Correlation IDs       OpenTelemetry     Chained Events
         |                 |                     |                 |
         v                 v                     v                 v
   [ Grafana ]       [ Log Aggregator ]    [ Sentry UI ]     [ Compliance ]
   5 Dashboards      ELK / CloudWatch      Zero-PII          SOC 2 Evidence
```

---

## 2. Pillar 1: Metrics (Prometheus & Grafana)
- Scrapes pull telemetry from `GET /metrics` every 15s.
- Custom application collectors monitor financial aggregates, AI pipeline durations, rate limit triggers, and outbox throughput.
- Grafana auto-provisions dashboards for platform health, finances, AI forensics, security, and outbox communications.

---

## 3. Pillar 2: Structured Logging & Correlation IDs
- All backend logs are formatted in structured JSON (`backend/app/core/logging.py`).
- Every HTTP request receives or inherits an `X-Correlation-ID` header, propagated through Python `contextvars` to all database operations, domain services, and outbound requests.
- When an issue occurs, operators search by correlation ID to reconstruct the exact request lifecycle.

---

## 4. Pillar 3: Distributed Tracing & Error Tracking (Sentry)
- Captures uncaught route handler exceptions and slow database queries.
- Traces spans across FastAPI routers, SQLAlchemy queries, and outbound HTTP calls.
- Strictly adheres to the Zero-PII sanitization policy.

---

## 5. Pillar 4: Tamper-Evident Security Audit Trail
- Synchronously records security-critical actions (`AUTH_LOGIN_SUCCESS`, `AUTH_LOCKOUT`, `USER_ROLE_CHANGED`, `ADMIN_ACTION`).
- Verifies integrity via sequential SHA-256 hash chaining, enabling automated non-repudiation verification.

