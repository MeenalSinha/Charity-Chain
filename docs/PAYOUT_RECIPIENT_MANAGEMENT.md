# NGO Payout Recipient & Destination Management
**CharityChain — Phase 8: Fund Disbursement & Financial Reconciliation**  
**Document Version:** 1.0.0  
**Date:** 2026-11-12

---

## 1. Overview

Payout destinations represent the verified accounts where charitable funds are disbursed to recipient organizations upon milestone completion.

Because destination accounts represent real financial endpoints, CharityChain enforces strict cryptographic protection, validation, and administrative verification workflows.

---

## 2. Security Architecture

### 2.1 Encryption at Rest (AES-GCM / Fernet)
- Sensitive destination details (PayPal email addresses, bank accounts) are never stored in plaintext within the database.
- Cipher is initialized using a 32-byte key derived from `settings.SECRET_KEY` via SHA-256:
  $$\text{Key} = \text{urlsafe\_b64encode}(\text{SHA-256}(\text{SECRET\_KEY}))$$
- Plaintext destination is encrypted into `encrypted_destination` prior to database persistence.

### 2.2 Blind Indexing via SHA-256 Hashing
- To enable fast indexing, querying, and duplicate prevention without decrypting records:
  $$\text{destination\_hash} = \text{SHA-256}(\text{normalized\_destination})$$
- Indexed uniquely per organization to prevent registering conflicting destinations.

### 2.3 Presentation Layer Masking
- The presentation layer and audit logs only expose masked strings:
  - Email: `finance@reforest.org` $\rightarrow$ `f***e@reforest.org`
  - Bank Account: `1234567890` $\rightarrow$ `****7890`
- Raw plaintext is exclusively accessible inside `PayoutRecipientService.get_decrypted_destination_for_payout` during genuine outbound payout batch construction.

---

## 3. Recipient Lifecycle State Machine

```
   ┌─────────────────────────────────────────────────────────┐
   │                     NGO Registration                    │
   │  (Format Validation -> Encryption at Rest -> Hash)      │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │    PENDING_VERIFICATION     │
                 └──────────────┬──────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼ (Verified by Admin/Operator)          ▼ (Rejected)
┌───────────────────────┐               ┌───────────────────────┐
│       VERIFIED        │               │       REJECTED        │
└───────────┬───────────┘               └───────────────────────┘
            │
            ├───────────────────────────────────────┐
            │                                       │
            ▼ (Suspicious Activity)                 ▼ (New Destination Added)
┌───────────────────────┐               ┌───────────────────────┐
│       SUSPENDED       │               │      is_primary       │
└───────────────────────┘               └───────────────────────┘
```

1. **`NOT_CONFIGURED`**: Organization has not registered any receiving accounts. Disbursements are blocked.
2. **`PENDING_VERIFICATION`**: Account registered, awaiting review by platform admin or finance operator.
3. **`VERIFIED`**: Account verified and authorized for milestone disbursements.
4. **`REJECTED`**: Account failed verification checks (name mismatch, fraud check).
5. **`SUSPENDED`**: Account temporarily frozen due to compliance or dispute concerns.
6. **`INACTIVE`**: Account retired by the NGO.

