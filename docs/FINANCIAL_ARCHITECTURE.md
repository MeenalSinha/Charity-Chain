# CharityChain Financial Architecture Specification
**Phase 8: Fund Disbursement & Financial Reconciliation**  
**Document Version:** 1.0.0  
**Date:** 2026-11-12  
**Status:** IMPLEMENTED & AUDITED

---

## 1. System Overview

CharityChain's financial architecture is engineered to provide absolute fiduciary integrity, non-repudiation, and mathematical transparency for charitable funds.

Unlike traditional platforms that rely on single-entry mutable balances or superficial escrow representations, CharityChain enforces:
1. **Mathematical Invariant Accounting:** Strict double-entry general ledger where $\sum \text{Debits} == \sum \text{Credits}$ for every committed transaction.
2. **Maker-Checker Segregation of Duties:** Outbound disbursement preparation is strictly separated from authorization ($Requester \neq Approver$). Self-approval is rejected at both API and service layers.
3. **Multi-Gate Milestone Eligibility:** Outbound funds can only be released after an approved milestone passes 9 compliance gates, including Phase 7 Dual-Key Human Review consensus and cryptographic evidence snapshot verification.
4. **Zero-Float Minor Unit Arithmetic:** All monetary values are processed and stored as integer minor units (cents), eliminating floating-point rounding errors.
5. **Real-Money Provider Integration:** Native outbound batch payouts via PayPal Payouts REST API v1 with idempotency deduplication (`sender_batch_id`) and automated settlement reconciliation.

---

## 2. High-Level Financial Domain Model

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     Donor Contributions                     │
   │  (PayPal Orders v2 -> Immediate Capture -> Minor Units)     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Double-Entry General Ledger Postings            │
   │  Debit 1000: Gateway Clearing (Asset)                       │
   │  Credit 2000: Designated Project Funds Liability (Liability)│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       │ (Phase 6 AI Verification + Phase 7 Human Dual-Key)  │
       ▼                                                     ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  Disbursement Eligibility    │              │  NGO Payout Recipient Mgmt   │
│  Multi-Gate Audit Engine     │              │  Encrypted at Rest (AES)     │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │    Maker-Checker Disbursement Lifecycle     │
               │  1. Maker: Finance Operator Requests        │
               │  2. Checker: Finance Approver Authorizes    │
               │  3. Ledger Commitment (Debit 2000, Cred 2100│
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │         Outbound Payout Batch Engine        │
               │  (PayPal Payouts Batch REST API v1)         │
               │  Deduplication via sender_batch_id          │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │     Automated Settlement Reconciliation     │
               │  Local Captures/Refunds/Disbursements <->   │
               │  External Gateway Settlement Reports        │
               └─────────────────────────────────────────────┘
```

---

## 3. Core Architectural Subsystems

### 3.1 Double-Entry General Ledger (`backend/app/services/ledger_service.py`)
- Every monetary flow produces atomic, balanced journal entries.
- Journal entries and lines are immutable. Corrections occur strictly through offsetting reversal entries.
- The trial balance calculator continuously verifies that total debits match total credits across the system.

### 3.2 Fund Allocation Engine (`backend/app/services/fund_allocation_service.py`)
- Provides real-time unencumbered balance calculations:
  $$\text{Available Balance} = \text{Gross Captures} - \text{Completed Refunds} - \text{Committed Payouts} - \text{Succeeded Payouts}$$
- Guarantees that funds reserved for approved disbursements cannot be double-spent or over-committed.

### 3.3 Maker-Checker Financial Governance (`backend/app/services/disbursement_service.py`)
- Strict segregation of duties:
  - `FINANCE_OPERATOR`: Initiates payout requests (Maker).
  - `FINANCE_APPROVER`: Audits eligibility and authorizes payouts (Checker).
  - `FINANCE_AUDITOR`: Read-only ledger and reconciliation access.
- Cryptographically logged authorization audit events (`PayoutApproval`).

### 3.4 Payout Recipient Onboarding (`backend/app/services/recipient_service.py`)
- Sensitive destination account information (e.g. PayPal email address, bank IBAN) is encrypted at rest using AES/Fernet derived from `SECRET_KEY`.
- Deterministic SHA-256 destination hashing for indexing and duplicate detection.
- UI masking (e.g., `f***e@cleanwaters.org`) prevents credential exposure in presentation layers.

### 3.5 Automated Settlement Reconciliation (`backend/app/services/reconciliation_service.py`)
- Daily or on-demand automated reconciliation jobs matching local records with external provider settlement files.
- Discrepancy taxonomy: `AMOUNT_MISMATCH`, `STATUS_MISMATCH`, `MISSING_IN_LOCAL`, `MISSING_IN_GATEWAY`, `FEE_MISMATCH`.
- Documented manual resolution workflow for audited exceptions.
