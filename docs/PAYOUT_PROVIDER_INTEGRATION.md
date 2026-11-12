# Outbound Payout Provider Integration Specification
**CharityChain — Phase 8: Fund Disbursement & Financial Reconciliation**  
**Document Version:** 1.0.0  
**Date:** 2026-11-12

---

## 1. Provider Architecture

CharityChain uses an extensible provider pattern (`backend/app/integrations/payouts/`) allowing pluggable outbound payment rails:
- `PayoutProviderInterface`: Abstract base class defining validation, batch submission, batch status polling, item status polling, and unclaimed cancellation.
- `PayPalPayoutProvider`: Production implementation interfacing with the PayPal Payouts REST API v1.
- `MockPayoutProvider`: In-memory deterministic mock with fault injection for automated testing.
- `PayoutProviderFactory`: Factory for provider resolution and runtime mock injection.

---

## 2. PayPal Payouts REST API v1 Specification

### 2.1 Authentication & Scopes
- **Grant Type:** `client_credentials`
- **Endpoint:** `POST {base_url}/v1/oauth2/token`
- **Scope Required:** `https://api-m.paypal.com/v1/payments/payouts`
- **Token Caching:** Thread-safe in-memory cache with safety buffer (`OAUTH_TOKEN_CACHE_TTL_SECONDS = 300s`).

### 2.2 Batch Creation (`POST /v1/payments/payouts`)
- **Headers:**
  - `Authorization: Bearer <access_token>`
  - `Content-Type: application/json`
- **Payload Schema:**
  ```json
  {
    "sender_batch_header": {
      "sender_batch_id": "BATCH-20261112120000-ABCDEF",
      "email_subject": "CharityChain milestone disbursement notice",
      "email_message": "Your milestone deliverables have been verified."
    },
    "items": [
      {
        "recipient_type": "EMAIL",
        "amount": {
          "value": "2500.00",
          "currency": "USD"
        },
        "note": "Milestone #1 sapling procurement payout",
        "sender_item_id": "disbursement-uuid-12345",
        "receiver": "finance@reforest.org"
      }
    ]
  }
  ```
- **Idempotency:** Enforced via `sender_batch_id`. Re-submitting an existing ID returns HTTP 409 or `DUPLICATE_BATCH_ID` and triggers `DuplicatePayoutBatchException`.

### 2.3 Status Polling & Webhooks
- **Batch Inspection:** `GET /v1/payments/payouts/{payout_batch_id}`
- **Item Inspection:** `GET /v1/payments/payouts-item/{payout_item_id}`
- **Webhook Events Ingested:**
  - `PAYMENT.PAYOUTSBATCH.SUCCESS`: Entire batch marked completed.
  - `PAYMENT.PAYOUTS-ITEM.SUCCEEDED`: Item marked `SUCCEEDED`, ledger execution posted.
  - `PAYMENT.PAYOUTS-ITEM.FAILED` / `BLOCKED`: Item marked `FAILED`, commitment reversed.
  - `PAYMENT.PAYOUTS-ITEM.UNCLAIMED` / `RETURNED`: Item marked `RETURNED`, commitment reversed.

---

## 3. Regulatory Corridors & India (IN) Restrictions

CharityChain's `PayoutCapabilityMatrix` (`capability.py`) evaluates currency and country feasibility:
- **India Domestic Restrictions (RBI / FEMA Compliance):**
  - Domestic INR payouts via the PayPal Payouts API are restricted by Reserve Bank of India regulations.
  - For Indian NGOs receiving foreign contributions (FCRA compliant), disbursements must be denominated in supported international currencies (USD/EUR/GBP) with compliant export invoicing, or settled via direct bank transfer rail.
- **Corridor Thresholds:**
  - USD: Min $1.00 (100 cents), Max Item $20,000.00 (2,000,000 cents), Max Batch $1,000,000.00.
  - EUR: Min €1.00, Max Item €18,000.00.
  - GBP: Min £1.00, Max Item £16,000.00.

