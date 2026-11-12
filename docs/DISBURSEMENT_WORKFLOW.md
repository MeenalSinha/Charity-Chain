# Milestone Disbursement & Payout Workflow
**CharityChain — Phase 8: Fund Disbursement & Financial Reconciliation**  
**Document Version:** 1.0.0  
**Date:** 2026-11-12

---

## 1. Overview

Disbursements in CharityChain are milestone-governed and event-driven. Outbound payouts cannot be initiated based on promises, schedules, or unverified claims. Funds are disbursed strictly following verified evidence and independent human review sign-off.

---

## 2. Multi-Gate Disbursement Eligibility Audit

Before any disbursement can be initiated, `DisbursementEligibilityService` audits 9 independent prerequisites:

| Gate # | Prerequisite | Verification Mechanism | Failure Consequence |
|---|---|---|---|
| **Gate 1** | Milestone Approved | `milestone.status in (APPROVED, COMPLETED)` | Payout Request Blocked |
| **Gate 2** | Project Active | `project.status in (ACTIVE, COMPLETED)` | Payout Request Blocked |
| **Gate 3** | Organization Verified | `org.status == OrganizationStatus.VERIFIED` | Payout Request Blocked |
| **Gate 4** | Human Review Case Approved | Latest `ReviewCase.final_decision == APPROVE` | Payout Request Blocked |
| **Gate 5** | Dual-Key Consensus | `reconciliation_result == APPROVED` or dual approved decisions | Payout Request Blocked |
| **Gate 6** | Evidence Digest Match | SHA-256 evidence snapshot digest matches `ReviewCase.snapshot_digest` | Tamper Alert / Blocked |
| **Gate 7** | No Open Disputes | All `ReviewDispute` records for milestone are `RESOLVED`, `CLOSED`, or `WITHDRAWN` | Payout Request Blocked |
| **Gate 8** | Verified Payout Recipient | Active `PayoutRecipient` in `VERIFIED` status | Payout Request Blocked |
| **Gate 9** | Fund Balance Sufficiency | `available_balance >= requested_amount` and no in-flight payout conflict | Payout Request Blocked |

---

## 3. Maker-Checker Segregation of Duties

To prevent fraud, collusion, and unauthorized release of funds, CharityChain enforces a strict Maker-Checker model:

### 3.1 Maker Role (`FINANCE_OPERATOR`)
- Initiates the disbursement request via `/api/v1/disbursements/request`.
- Evaluates eligibility report and specifies requested amount within the available balance ceiling.
- System transitions status to `PENDING_APPROVAL` and logs `PayoutApproval(action="REQUESTED")`.

### 3.2 Checker Role (`FINANCE_APPROVER`)
- Reviews the disbursement request via the Financial Operations Center.
- Audits eligibility report, evidence review findings, and unencumbered fund balance.
- **Enforced Constraint:** If `approver_id == requested_by_id`, the system raises `ForbiddenException`. Self-approval is rejected.
- Upon approval:
  1. Status transitions to `APPROVED`.
  2. Double-entry commitment journal entry is posted:
     - Debit `2000: Designated Project Funds Liability`
     - Credit `2100: Committed Milestone Payouts Payable`
  3. Action logged in `PayoutApproval(action="APPROVED")`.

---

## 4. Payout Batching & Gateway Execution

1. **Batch Assembly:** The Finance Approver selects one or more `APPROVED` disbursements and triggers batch submission.
2. **Idempotency Guarantee:** A unique `sender_batch_id` (`BATCH-{YYYYMMDDHHMMSS}-{UUID}`) is generated. Duplicate submissions are rejected.
3. **Gateway Dispatch:** Payload is delivered to PayPal REST API `POST /v1/payments/payouts`.
4. **Immediate Status Handling:**
   - Synchronous Success: Status updated to `SUCCEEDED`, and ledger execution entry posted:
     - Debit `2100: Committed Milestone Payouts Payable`
     - Credit `1000: Gateway Clearing - PayPal`
     - Debit `5000: Payment Processing Fees Expense` (if fee assessed)
   - Synchronous Failure: Status updated to `FAILED`, and commitment reversed:
     - Debit `2100: Committed Milestone Payouts Payable`
     - Credit `2000: Designated Project Funds Liability`
5. **Asynchronous Webhook Settlement:** As PayPal processes individual items, webhook notifications update item statuses and complete ledger postings.
