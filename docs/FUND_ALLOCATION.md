# Fund Allocation Architecture & Accounting Specification

## 1. Executive Overview

CharityChain enforces strict financial segregation, mathematical immutability, and deterministic allocation of donor capital across project milestones. Donors fund specific charitable projects, and funds are reserved for and allocated to specific, verifiable milestones according to project budgets and verified completion gates.

> [!IMPORTANT]
> **Legal & Custodial Disclosure**:
> Internal milestone fund allocations and ledger accounts (e.g., `20100 - Project Milestone Allocation Liability`) represent internal accounting reservations and platform obligations. They do **not** constitute legally chartered banking escrow trusts, fiduciary custodial accounts, or bankruptcy-remote funds. Only confirmed provider payouts constitute actual disbursement. Completed milestones and internal approvals do not guarantee fund disbursement until an authorized provider-confirmed payout occurs.

To ensure absolute auditability and prevent fractional rounding discrepancies, **all monetary calculations strictly operate on 64-bit integer minor currency units (cents)**. Floating-point arithmetic is forbidden across the entire financial subsystem.

---

## 2. Core Principles of Fund Allocation

1. **Integer Minor Unit Currency Representation**:
   - Amounts are represented and stored as integers in the smallest denomination (e.g., USD cents, EUR cents: `10000` = `$100.00`).
   - Database columns: `BigInteger` for account balances, journal entries, and allocations.
2. **Gross vs. Net Donation Accounting**:
   - Donors initiate donations with a **Gross Amount** ($A_{gross}$).
   - Payment gateways deduct processing fees ($F_{proc}$).
   - **Net Amount** credited to the Project Escrow:
     $$A_{net} = A_{gross} - F_{proc}$$
   - Only settled and completed net funds ($A_{net}$) can be allocated or disbursed. Pending, failed, or disputed charges are excluded from allocatable funds.
3. **Milestone Segregation**:
   - Each project contains ordered milestones ($M_1, M_2, \dots, M_k$), each defined with a target budget ($T_i$).
   - Total project target: $T_{proj} = \sum_{i=1}^k T_i$.
   - Milestone allocation ensures funds cannot be commingled or diverted between separate milestones without formal governance reallocation.
4. **Conservation of Value Invariant**:
   At any point in time, for any project:
   $$\text{Total Net Funds Received} = \text{Unallocated Escrow} + \sum_{i=1}^k (\text{Disbursed}_i + \text{Pending}_i + \text{Remaining Allocated}_i)$$

---

## 3. Mathematical Allocation Model

### 3.1 Balance Definitions

For any project milestone $M_i$:
- **Target Budget ($T_i$)**: The target amount required to complete milestone $i$.
- **Committed Funds ($C_i$)**: Total capital reserved and allocated to milestone $i$.
- **Disbursed Funds ($D_i$)**: Total funds successfully paid out to the NGO payout recipient for milestone $i$ (status = `COMPLETED`).
- **Pending Disbursements ($P_i$)**: Funds currently locked in the disbursement pipeline (status $\in \{\text{REQUESTED}, \text{APPROVED}, \text{BATCHED}, \text{PROCESSING}\}$).
- **Available Milestone Balance ($B_{avail, i}$)**:
  $$B_{avail, i} = C_i - (D_i + P_i)$$

### 3.2 Allocation Ceiling

Under no circumstance may a disbursement request $A_{req}$ for milestone $i$ exceed the available milestone balance:
$$A_{req} \le B_{avail, i}$$
If $A_{req} > B_{avail, i}$, the `DisbursementEligibilityService` immediately trips Gate 5 (`INSUFFICIENT_FUNDS`), blocking disbursement generation.

### 3.3 Dynamic Milestone Funding

When donations arrive for a project:
1. Donated funds are deposited into the **Project Escrow Account** in the double-entry ledger.
2. In proportional or sequential allocation modes:
   - Sequential: Milestone $M_1$ is funded up to $T_1$, then $M_2$ up to $T_2$, etc.
   - Proportional: Funds are apportioned according to each milestone's budget proportion $w_i = \frac{T_i}{T_{proj}}$, rounding remaining cents to the earliest unsatisfied milestone.

---

## 4. Fund Allocation Service Implementation

The `FundAllocationService` (`backend/app/services/fund_allocation_service.py`) provides real-time balance inquiries and validation:

```python
class FundAllocationService:
    @staticmethod
    def get_milestone_balance(db: Session, milestone_id: UUID) -> MilestoneFinancialBalance:
        ...
```

### Metrics Returned:
- `milestone_id`: Unique identifier of the milestone.
- `project_id`: Parent project identifier.
- `target_amount_cents`: Planned budget in cents.
- `allocated_amount_cents`: Total net funding allocated to this milestone.
- `disbursed_amount_cents`: Total settled disbursements.
- `pending_amount_cents`: Total in-flight disbursements.
- `available_balance_cents`: Remaining disbursable funds.
- `currency`: ISO 4217 3-letter currency code (e.g., `USD`).

---

## 5. Ledger Integration

Every change in fund allocation corresponds to journal entries in the double-entry ledger:
- **Donation Received**:
  - Debit: `10100 - Gateway Settlement Asset` (Gross)
  - Credit: `50100 - Payment Processor Expense` (Gateway Fee)
  - Credit: `20100 - Project Escrow Liability` (Net)
- **Disbursement Approved/Locked**:
  - Debit: `20100 - Project Escrow Liability` (Disbursement Amount)
  - Credit: `20200 - Disbursement Clearing Liability` (Disbursement Amount)
- **Disbursement Settled via Payout**:
  - Debit: `20200 - Disbursement Clearing Liability` (Disbursement Amount)
  - Credit: `10200 - Outbound Payout Clearing Asset` (Disbursement Amount)
  - Debit: `50200 - Payout Provider Fee Expense` (Payout Fee)
  - Credit: `10200 - Outbound Payout Clearing Asset` (Payout Fee)

---

## 6. Edge Cases & Resilience

1. **Refunds & Chargebacks**:
   - If a donation is refunded, the gross and fee amounts are reversed.
   - If unallocated escrow is insufficient to cover the refund, affected un-disbursed milestone allocations are reduced in reverse chronological order, logging high-priority audit events.
2. **Milestone Rejection or Cancellation**:
   - If a milestone is rejected by human reviewers or cancelled, remaining committed funds ($B_{avail, i}$) are returned to the project's unallocated escrow pool for reallocation or donor return.
3. **Multi-Currency Isolation**:
   - CharityChain currently pins projects to single base currencies (default: `USD`).
   - Cross-currency donations are converted at gateway capture time and recorded in the project base currency.

