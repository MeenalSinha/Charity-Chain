# CharityChain — External Webhook Subscriptions & Integration Guide

## 1. Overview

CharityChain enables partner NGOs, auditing bodies, and donor platforms to receive real-time, event-driven HTTP POST notifications as milestones, verifications, payments, and certificates are processed.

---

## 2. Webhook Subscription Lifecycle

```
[ POST /api/v1/webhooks/subscriptions ]
                 │
                 ├── SSRF Target URL Validation
                 ├── Generate Random HMAC Secret (whsec_...)
                 ├── Encrypt Secret at Rest (Fernet AES-256)
                 ▼
[ Return Subscription ID + Plaintext Secret (ONCE) ]
                 │
                 ▼
[ Subscription ACTIVE ] ──(10 Consecutive Failures)──► [ SUSPENDED ]
                 │                                            │
                 ├── Rotate Secret                            ├── Fix Endpoint
                 │                                            ▼
                 ▼                                     [ Re-activate ]
[ DELETE /api/v1/webhooks/subscriptions/{id} ]
                 │
                 ▼
[ Cascaded Deletion of Secrets & Delivery Logs ]
```

---

## 3. Subscription Management Endpoints

All webhook endpoints are located under `/api/v1/webhooks/subscriptions` and require NGO Owner/Admin or Platform Admin permissions.

### `POST /api/v1/webhooks/subscriptions`
Registers a new outbound webhook endpoint.

- **Request Body**:
```json
{
  "endpoint_url": "https://partner-ngo.org/api/charitychain-events",
  "description": "Primary Production Webhook",
  "subscribed_events": [
    "payment.captured",
    "certificate.issued",
    "disbursement.completed"
  ]
}
```

- **Response (201 Created)**:
```json
{
  "subscription": {
    "id": "sub-11223344",
    "endpoint_url": "https://partner-ngo.org/api/charitychain-events",
    "description": "Primary Production Webhook",
    "status": "ACTIVE",
    "subscribed_events": ["payment.captured", "certificate.issued", "disbursement.completed"],
    "created_at": "2026-11-12T22:30:00Z"
  },
  "secret": "whsec_9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"
}
```

> [!WARNING]
> The plaintext HMAC secret `whsec_...` is returned **only once** at creation. The platform encrypts the secret using AES-256 and never displays it again. Store it securely in your secret manager.

---

### `POST /api/v1/webhooks/subscriptions/{subscription_id}/rotate-secret`
Rotates the signing secret for a subscription. Immediately deactivates the prior secret, provisions a new secret, and returns the new plaintext key once.

---

### `DELETE /api/v1/webhooks/subscriptions/{subscription_id}`
Permanently deletes a subscription and cascades removal to all delivery attempt records and secrets.

---

### `GET /api/v1/webhooks/subscriptions/{subscription_id}/deliveries`
Returns delivery attempt history, HTTP status codes, latency in milliseconds, and error messages for auditing and troubleshooting.

