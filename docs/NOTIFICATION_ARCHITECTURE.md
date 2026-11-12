# CharityChain — Notification & Communication Architecture

## 1. Executive Summary

Phase 10 introduces a production-grade, distributed transactional communication infrastructure for **CharityChain**. The architecture decouples domain state mutations from external communication side-effects through the **Transactional Outbox Pattern**, ensuring guaranteed at-least-once delivery, zero data loss, and absolute isolation between financial accounting and external transport failures.

---

## 2. Architectural Principles

1. **Transactional Integrity (Outbox Pattern)**:
   Domain services (Donations, Evidence, Reviews, Ledger, Disbursements, Certificates) write domain state changes and corresponding `DomainEvent` records within the same atomic database transaction. No external network calls (SMTP, HTTP Webhooks) occur inside the primary transaction boundary.

2. **Decoupled Asynchronous Dispatch**:
   An independent background worker (`NotificationWorker`) polls the `notification_outbox` table, batches pending events, generates in-app alerts, renders responsive transactional emails, and signs outbound webhooks.

3. **Multi-Channel Delivery**:
   - **In-App Notification Center**: User-isolated alerts with real-time unread badges, read/archive states, and granular preference controls.
   - **Transactional Email**: Branded, responsive HTML layouts with plain-text fallbacks, strict XSS escaping, and dual-provider architecture (SMTP with STARTTLS/SSL and DevNotificationSink).
   - **Outbound Webhooks**: HMAC-SHA256 cryptographic signing, timestamp-based replay protection, SSRF defense engine, exponential backoff with full jitter, and auto-suspension thresholds.

4. **Fail-Closed Security**:
   - SSRF protection blocks private RFC 1918 subnets, loopback interfaces, cloud metadata endpoints, and dangerous service ports.
   - Webhook signing secrets are encrypted at rest using Fernet AES-256 and disclosed only once upon creation.
   - Mandatory security alerts cannot be disabled by user communication preferences.
   - DevSink and plain HTTP webhooks are strictly prohibited in production environments.

---

## 3. High-Level System Architecture

```
[ Domain Operation ]
(Donation Capture, Review Decision, Certificate Minting, Disbursement)
        │
        ▼ (Atomic DB Transaction)
┌────────────────────────────────────────────────────────┐
│  Primary Database (PostgreSQL / SQLite)                │
│  ├── Domain Tables (donations, certificates, etc.)     │
│  └── notification_outbox                               │
│      ├── event_id (UUID)                               │
│      ├── event_type (e.g. payment.captured)            │
│      ├── payload_json (RFC 8785 Canonical JSON)        │
│      └── status (PENDING / PROCESSING / PROCESSED)     │
└────────────────────────────────────────────────────────┘
        │
        │ (Poll / Claim Batch)
        ▼
┌────────────────────────────────────────────────────────┐
│  NotificationWorker (Background Engine)                │
│  ├── Idempotency & Batch Coordination                  │
│  ├── Recipient Resolution                              │
│  └── Preference & Channel Filtering                    │
└───────┬───────────────────────┬────────────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐   ┌───────────────────┐   ┌─────────────────────┐
│  In-App Alerts  │   │ Transactional     │   │ Outbound Webhooks   │
│  notifications  │   │ Email Provider    │   │ Dispatcher          │
│  table          │   │ (SMTP / DevSink)  │   │ (HMAC-SHA256 + SSRF)│
└─────────────────┘   └───────────────────┘   └─────────────────────┘
```

---

## 4. Database Entities & Relationships

| Model | Table Name | Purpose |
|---|---|---|
| `NotificationOutbox` | `notification_outbox` | Durable outbox event records with retry tracking |
| `Notification` | `notifications` | In-app user notification center alerts |
| `NotificationPreference` | `notification_preferences` | User-configured channel and topic communication preferences |
| `NotificationDeliveryAttempt`| `notification_delivery_attempts`| Detailed telemetry for email delivery attempts |
| `WebhookSubscription` | `webhook_subscriptions` | External endpoint registrations with event filters |
| `WebhookEndpointSecret` | `webhook_endpoint_secrets` | Fernet-encrypted HMAC signing secrets with rotation history |
| `WebhookDelivery` | `webhook_deliveries` | Durable event delivery records per subscription |
| `WebhookDeliveryAttempt`| `webhook_delivery_attempts`| HTTP status codes, latency, and error logs per dispatch attempt |
| `NotificationTemplateVersion`| `notification_template_versions`| Versioned notification content and schema definitions |

---

## 5. Worker Lifecycle & Scalability

The `NotificationWorker` can run in multiple operational modes:
- **Continuous Daemon**: `python -m backend.app.cli run-notification-worker --poll-interval 5.0 --batch-size 50`
- **Cron / One-Shot**: `python -m backend.app.cli run-notification-worker --once`
- **Embedded Application Mode**: Invoked on scheduled intervals within container environments.

The worker claims events using status transitions (`PENDING` -> `PROCESSING` -> `PROCESSED`), preventing race conditions across concurrent worker instances through transactional database locks.

