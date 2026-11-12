# Double-Entry Accounting Ledger Specification
**CharityChain — Phase 8: Fund Disbursement & Financial Reconciliation**  
**Document Version:** 1.0.0  
**Date:** 2026-11-12

---

## 1. Accounting Principles

CharityChain enforces the classical accounting equation across all charitable operations:
$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

For every financial movement within the system:
$$\sum \text{Debits} == \sum \text{Credits}$$

Any attempt to commit an unbalanced transaction, record negative minor units, or alter existing journal lines is rejected with `ValidationException`.

---

## 2. Standard Chart of Accounts

| Account Number | Account Name | Type | Normal Balance | Purpose |
|---|---|---|---|---|
| **1000** | Gateway Clearing - PayPal | **ASSET** | Debit | Cash and unsettled funds held at processor gateways. |
| **2000** | Designated Project Funds Liability | **LIABILITY** | Credit | Fiduciary obligation for donor funds designated to active milestones. |
| **2100** | Committed Milestone Payouts Payable | **LIABILITY** | Credit | Authorized milestone disbursements awaiting gateway settlement. |
| **4000** | Platform Operations Revenue | **REVENUE** | Credit | Unrestricted operational contributions and general donations. |
| **5000** | Payment Processing Fees Expense | **EXPENSE** | Debit | Direct transaction processing charges incurred by gateways. |

---

## 3. Transaction Journal Mapping

### 3.1 Donation Capture
*Trigger:* Donor contribution successfully captured via PayPal Orders v2.
- **Debit `1000` (Gateway Clearing):** Gross captured amount (\$500.00 / 50,000 cents).
- **Credit `2000` (Designated Project Funds Liability):** Gross captured amount (\$500.00 / 50,000 cents).
- *If processor fee assessed (\$15.00 / 1,500 cents):*
  - **Debit `5000` (Processing Fees Expense):** \$15.00
  - **Credit `1000` (Gateway Clearing):** \$15.00

### 3.2 Donor Refund
*Trigger:* Partial or full refund confirmed as `COMPLETED`.
- **Debit `2000` (Designated Project Funds Liability):** Refund amount (\$100.00).
- **Credit `1000` (Gateway Clearing):** Refund amount (\$100.00).

### 3.3 Milestone Payout Commitment (Maker-Checker Authorization)
*Trigger:* Independent Finance Approver signs off on milestone disbursement request.
- **Debit `2000` (Designated Project Funds Liability):** Authorized disbursement amount (\$3,000.00).
- **Credit `2100` (Committed Milestone Payouts Payable):** Authorized disbursement amount (\$3,000.00).

### 3.4 Payout Execution & Settlement
*Trigger:* Outbound payout batch item confirmed as `SUCCEEDED`.
- **Debit `2100` (Committed Milestone Payouts Payable):** Payout amount (\$3,000.00).
- **Credit `1000` (Gateway Clearing):** Payout amount (\$3,000.00).
- *If payout fee assessed (\$0.25 / 25 cents):*
  - **Debit `5000` (Processing Fees Expense):** \$0.25
  - **Credit `1000` (Gateway Clearing):** \$0.25

### 3.5 Payout Failure Reversal
*Trigger:* Payout batch or item fails, is blocked, or returned unclaimed.
- **Debit `2100` (Committed Milestone Payouts Payable):** Amount.
- **Credit `2000` (Designated Project Funds Liability):** Amount (restored to available funds).

---

## 4. Immutability & Reversals

- Journal entries and lines are write-only. Database triggers and service rules prohibit `UPDATE` or `DELETE`.
- If an erroneous transaction occurs, an authorized operator posts an audited reversal entry via `reverse_journal_entry(...)`.
- The original entry is marked `is_reversed = True` with a foreign key pointer `reversed_by_entry_id` linking to the offsetting journal entry.
- Re-reversing an entry is rejected with `ConflictException`.
