# PayPal Payment Gateway Integration Architecture

## 1. Overview & Architectural Scope

CharityChain integrates directly with the **PayPal REST API Orders v2** and **Payments v2** services. The integration replaces static pledging with genuine payment execution, ensuring every contribution is backed by an authentic gateway order and settled capture.

The payment gateway layer is completely abstracted through `PaymentProviderInterface` (`backend/app/integrations/payments/base.py`), with the production implementation delivered by `PayPalPaymentProvider` (`backend/app/integrations/payments/paypal.py`) and high-fidelity testing powered by `MockPayPalProvider` (`tests/mocks/paypal_mock.py`).

```
+-----------------------------------------------------------------------------------+
|                                  CharityChain Core                                |
|                                                                                   |
|  [Donor Client / Streamlit]                                                       |
|             |                                                                     |
|             v (Create Intent)                                                     |
|  [POST /api/v1/donations]                                                         |
|             |                                                                     |
|             v                                                                     |
|   DonationService                                                                 |
|   - Eligibility Validation                                                        |
|   - Money Minor Units Formatting                                                  |
|   - Donation (CREATED)                                                            |
|             |                                                                     |
|             v                                                                     |
|   PaymentProviderInterface                                                        |
|             |                                                                     |
|    +--------+-------------------------------------+                               |
|    |                                              |                               |
|    v                                              v                               |
| PayPalPaymentProvider                    MockPayPalProvider (Tests)               |
| - OAuth 2.0 Bearer Token Management      - Deterministic State                    |
| - Exponential Backoff Retries            - Configurable Gateway Failures          |
| - Strict Timeout Controls                - In-Memory Order/Capture Store          |
|    |                                                                              |
+----+------------------------------------------------------------------------------+
     |
     v HTTPS (TLS 1.2+)
+-----------------------------------------------------------------------------------+
|                             PayPal REST API Endpoints                             |
|                                                                                   |
| - POST /v1/oauth2/token                    (OAuth 2.0 Client Credentials)        |
| - POST /v2/checkout/orders                 (Create Order intent=CAPTURE)          |
| - GET  /v2/checkout/orders/{id}            (Get Order Details)                    |
| - POST /v2/checkout/orders/{id}/capture    (Execute Capture)                      |
| - GET  /v2/payments/captures/{id}          (Get Settled Capture)                  |
| - POST /v2/payments/captures/{id}/refund   (Execute Partial/Full Refund)          |
| - POST /v1/notifications/verify-webhook... (Cryptographic Signature Verification) |
+-----------------------------------------------------------------------------------+
```

---

## 2. OAuth 2.0 Token Management & Security

`PayPalPaymentProvider` implements client credentials grant authentication compliant with RFC 6749:

- **Token Acquisition**: Requests access tokens from `/v1/oauth2/token` using HTTP Basic Authentication (`PAYPAL_CLIENT_ID:PAYPAL_CLIENT_SECRET` encoded in Base64).
- **In-Memory Caching**: Tokens are cached in-memory with automatic expiration tracking.
- **Expiration Safety Buffer**: The provider deducts `OAUTH_TOKEN_CACHE_TTL_SECONDS` (configured by default to 300 seconds) from the gateway's `expires_in` return value. This ensures no request is dispatched with an expired or nearly expired Bearer token.
- **Thread Concurrency Protection**: Token acquisition and refresh are synchronized with a `threading.Lock()` to prevent race conditions or multiple concurrent refresh calls.
- **Auto-Invalidation**: If any subsequent REST call encounters an HTTP 401 or 403 status code, the cached token is immediately purged, forcing a fresh token acquisition on the next cycle.

---

## 3. PayPal Orders v2 Contract: Intent CAPTURE

CharityChain enforces the `CAPTURE` intent for all standard charitable contributions. 

### Why Intent CAPTURE over AUTHORIZE?
PayPal pre-authorizations have a strict 29-day validity window and incur authorization fees on merchant accounts. Projects on CharityChain typically execute across multiple weeks or months before milestone review. Therefore:
1. Donated funds are immediately captured into the platform's PayPal balance.
2. Funds are strictly held in platform accounts and marked as project/milestone funding aggregates.
3. No payouts or disbursements to NGOs are performed in Phase 4 (disbursements are gated until human evidence review in Phase 8).

### Order Payload Structure
```json
{
  "intent": "CAPTURE",
  "purchase_units": [
    {
      "reference_id": "<donation_uuid>",
      "custom_id": "<donation_uuid>",
      "description": "Donation - <donation_uuid>",
      "amount": {
        "currency_code": "USD",
        "value": "25.00"
      }
    }
  ],
  "application_context": {
    "brand_name": "CharityChain",
    "landing_page": "BILLING",
    "user_action": "PAY_NOW",
    "return_url": "https://charitychain.org/donations/return",
    "cancel_url": "https://charitychain.org/donations/cancel"
  }
}
```

---

## 4. Payment Execution & Capture

1. **Donor Approval**: The donor is redirected to the gateway approval URL (`https://www.sandbox.paypal.com/checkoutnow?token=ORD-...`).
2. **Return & Capture**:
   - The donor approves the transaction and is redirected back to the platform return URL.
   - The frontend / client dispatches `POST /api/v1/payments/orders/{order_identifier}/capture`.
   - The backend queries `/v2/checkout/orders/{id}` to verify that PayPal status is `APPROVED` or `COMPLETED`.
   - The backend validates that the gateway order currency and amount match the database record down to the cent.
   - The backend invokes `/v2/checkout/orders/{id}/capture` with a unique idempotency key (`PayPal-Request-Id`).
   - The settled capture ID is stored in `PaymentCapture`, the order marked `COMPLETED`, the donation marked `PAID`, and raised funds incremented.

---

## 5. Refunds via Payments v2 API

Refunds are executed against settled captures using `POST /v2/payments/captures/{capture_id}/refund`:
- Supports both partial and full refunds.
- Validates that the requested refund does not exceed `remaining_refundable_balance = capture_amount - sum(completed_refunds)`.
- Updates the donation status to `PARTIALLY_REFUNDED` or `REFUNDED`.
- Decrements the project and milestone `raised_amount_minor_units` in real time.

---

## 6. Resilience, Retries & Error Handling

- **Connection & Read Timeouts**: Controlled via `PAYPAL_CONNECT_TIMEOUT` (10s) and `PAYPAL_READ_TIMEOUT` (30s).
- **Idempotent Retry Policy**: Idempotent operations (such as `GET` requests) are retried up to `PAYPAL_RETRY_LIMIT` (3 times) with exponential backoff on transient 5xx errors or network timeouts.
- **Non-Idempotent Safety**: Financial state mutations (`POST /v2/checkout/orders`, `POST /v2/checkout/orders/{id}/capture`, `POST /v2/payments/captures/{id}/refund`) are **never** blindly retried without an idempotency key.
- **Typed Error Translation**: PayPal HTTP errors are translated into domain exceptions:
  - 401/403 $\rightarrow$ `PaymentAuthenticationException`
  - 404 $\rightarrow$ `PaymentOrderNotFoundException`
  - 422 $\rightarrow$ `PaymentProviderException` (with parsed issue details)
  - 5xx / timeouts $\rightarrow$ `PaymentProviderException`
