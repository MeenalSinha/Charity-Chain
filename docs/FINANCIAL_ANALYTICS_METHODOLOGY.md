# CharityChain — Financial Analytics & Reconciliation Methodology

## 1. Principles of Financial Integrity
In charitable platforms, financial reporting must be completely unambiguous, mathematically verifiable, and resilient to inconsistencies. Phase 11 standardizes financial analytics on three architectural foundations:
1. **Integer Minor Units**: All monetary values are represented and stored as 64-bit integers representing cents/minor units (e.g. `$100.00` = `10000`). Float arithmetic is strictly prohibited in financial calculations.
2. **Authoritative Gateway State**: Analytics reflect actual settlement states confirmed via payment gateways (PayPal captures and refunds) and disbursement payouts.
3. **Double-Entry Ledger Consistency**: The internal general ledger (`journal_entries` and `journal_lines`) is verified using standard trial balance consistency checks.

---

## 2. Core Financial Metric Formulations

### 2.1 Gross Captured Donations
$$\text{Captured Minor Units} = \sum \text{PaymentCapture.amount\_minor\_units} \quad \text{where} \quad \text{status} = \text{PaymentCaptureStatus.COMPLETED}$$

### 2.2 Processed Refunds
$$\text{Refunded Minor Units} = \sum \text{PaymentRefund.amount\_minor\_units} \quad \text{where} \quad \text{status} = \text{PaymentRefundStatus.COMPLETED}$$

### 2.3 Net Funding Raised
$$\text{Net Funding} = \max(0, \text{Captured Minor Units} - \text{Refunded Minor Units})$$

### 2.4 Confirmed Disbursements
$$\text{Disbursed Minor Units} = \sum \text{Disbursement.amount\_minor\_units} \quad \text{where} \quad \text{status} = \text{DisbursementStatus.SUCCEEDED}$$

### 2.5 Escrow Balance
$$\text{Escrow Minor Units} = \max(0, \text{Net Funding} - \text{Disbursed Minor Units})$$

---

## 3. Double-Entry General Ledger Trial Balance
The general ledger enforces double-entry bookkeeping:
$$\sum \text{JournalLine.debit\_minor\_units} = \sum \text{JournalLine.credit\_minor\_units}$$

The `PlatformAnalyticsService.get_financial_operations_analytics` method executes a live integrity check:
```python
total_debits = db.scalar(select(func.coalesce(func.sum(JournalLine.debit_minor_units), 0))) or 0
total_credits = db.scalar(select(func.coalesce(func.sum(JournalLine.credit_minor_units), 0))) or 0
discrepancy = abs(total_debits - total_credits)
is_balanced = (discrepancy == 0)
```
If `discrepancy > 0`, the platform immediately flags the imbalance in administrator health dashboards and telemetry alerts.

---

## 4. Multi-Currency Segregation
CharityChain treats currencies as distinct asset buckets:
- Multi-currency donations are **never** summed into a single flat value.
- Portfolios and financial reports return an associative map broken down by ISO currency code:
  ```json
  {
    "USD": { "captured_cents": 250000, "refunded_cents": 5000, "net_cents": 245000 },
    "EUR": { "captured_cents": 120000, "refunded_cents": 0, "net_cents": 120000 }
  }
  ```
- Summary cards display the primary active currency with clear currency ISO indicators.

---

## 5. Audit & Reconciliation Cycle
Automated background jobs reconcile local captures against PayPal settlement batches:
- Matched transactions are confirmed in the ledger.
- Unmatched or discrepant records create `ReconciliationItem` records with status `DISCREPANCY`.
- Unresolved discrepancies are surfaced in the Platform Operations dashboard until manually resolved by finance operators.

