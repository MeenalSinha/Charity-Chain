# PayPal Webhook Notification Architecture & Processing

## 1. Webhook Endpoint Specification

- **Path**: `POST /api/v1/webhooks/paypal`
- **Authentication**: Cryptographic Asymmetric Signature Verification via PayPal Notifications API v1
- **Payload Format**: JSON (`application/json`)
- **HTTP Status Codes**:
  - `200 OK`: Webhook successfully verified and processed (or duplicate acknowledged)
  - `400 Bad Request`: Cryptographic signature verification failed or missing required headers
  - `500 Internal Server Error`: Unhandled database or system exception

---

## 2. Cryptographic Signature Verification Flow

Incoming PayPal webhook payloads are cryptographically signed using asymmetric keys. CharityChain verifies each request using the PayPal REST API `/v1/notifications/verify-webhook-signature`:

```
[Incoming HTTP POST /api/v1/webhooks/paypal]
       │
       ▼ Extract Headers:
       - PAYPAL-AUTH-ALGO
       - PAYPAL-CERT-URL (Must be valid *.paypal.com domain)
       - PAYPAL-TRANSMISSION-ID
       - PAYPAL-TRANSMISSION-SIG
       - PAYPAL-TRANSMISSION-TIME
       │
       ▼ Check Required Headers Present
       │ (If missing -> 400 Bad Request)
       │
       ▼ Call PayPal REST API:
       POST /v1/notifications/verify-webhook-signature
       {
         "auth_algo": "...",
         "cert_url": "...",
         "transmission_id": "...",
         "transmission_sig": "...",
         "transmission_time": "...",
         "webhook_id": "<CONFIGURED_WEBHOOK_ID>",
         "webhook_event": <RAW_BODY_JSON>
       }
       │
       ▼ Verification Result:
       - If SUCCESS: Proceed to processing
       - If FAILURE: Log security warning, record failed event, return 400
```

### Signature Replay & Header Spoofing Protection
- In production, signature verification is **strictly mandatory** (`PAYPAL_WEBHOOK_VERIFY_SIGNATURE=True`).
- Unverified requests cannot trigger any financial state transitions or database modifications.

---

## 3. Webhook Event Deduplication Strategy

PayPal provides at-least-once webhook delivery, meaning identical events may be delivered multiple times. CharityChain enforces strict deduplication through the dedicated `webhook_events` database table:

```sql
CREATE TABLE webhook_events (
    id VARCHAR(36) PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    event_id VARCHAR(100) NOT NULL UNIQUE,  -- Gateway Event ID deduplication key
    event_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(100),
    resource_type VARCHAR(100),
    signature_verification_status VARCHAR(50) NOT NULL,
    processing_status VARCHAR(50) NOT NULL,
    payload TEXT,
    failure_reason TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

### Deduplication Execution Logic
1. Upon receipt, the processor queries `SELECT * FROM webhook_events WHERE event_id = :event_id`.
2. If an event is already present with `processing_status = 'PROCESSED'`:
   - The event is immediately acknowledged with HTTP 200 and response `{"status": "DUPLICATE", "event_id": event_id}`.
   - No database entities are modified.
3. If the event is new, it is inserted with `processing_status = 'PENDING'`.
4. Upon successful business logic execution, status is transitioned to `PROCESSED`.
5. If processing fails, status is transitioned to `FAILED` with detailed diagnostic error logs.

---

## 4. Supported Event Types & Business Actions

| Event Type | Resource Type | Action Executed by CharityChain |
| :--- | :--- | :--- |
| `CHECKOUT.ORDER.APPROVED` | `checkout-order` | Automatically invokes `capture_donation_payment()` if not yet captured |
| `PAYMENT.CAPTURE.COMPLETED` | `capture` | Verifies matching `PaymentOrder` and ensures `Donation` is marked `PAID` |
| `PAYMENT.CAPTURE.DENIED` | `capture` | Marks local capture as `DECLINED` and donation as `FAILED` |
| `PAYMENT.CAPTURE.REFUNDED` | `refund` | Records external refund event and verifies funding totals are decremented |

