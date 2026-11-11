# Financial Data Integrity & Separation of Accounts

## 1. Zero Floating-Point Arithmetic Policy

CharityChain enforces an absolute ban on IEEE 754 floating-point representations (`float`, `double`) for monetary values. Floating-point arithmetic introduces rounding and representation errors (e.g. `0.1 + 0.2 = 0.30000000000000004`) that compromise accounting ledger integrity.

### Money Value Object Pattern
All financial quantities in CharityChain are represented via the `Money` immutable value object:
- **Internal Storage**: Exact integers representing currency minor units (e.g. cents, pence, paise, yen).
- **Type**: `int` in Python, `Integer` / `BigInteger` in database columns.
- **Conversion to Gateway Format**: Converted to exact string representations only at the external API boundary using explicit currency decimal scales (`CURRENCY_MINOR_UNITS`):
  - USD, EUR, GBP, CAD, AUD, CHF, INR: Scale 2 (e.g. `5000 minor units` $\rightarrow$ `"50.00"`)
  - JPY: Scale 0 (e.g. `5000 minor units` $\rightarrow$ `"5000"`)

---

## 2. Platform Account Holding vs. NGO Disbursements

A critical design requirement of CharityChain is the strict operational separation between **donation collection** and **NGO payouts**:

1. **Donation Collection (Phase 4)**:
   - When a donor contributes through PayPal, funds are captured into CharityChain's central platform merchant balance.
   - The platform acts as the collection agent and holds all received funds.
   - Projects and milestones reflect *committed impact funding*, not disbursed cash.
2. **Disbursement Gating (Deferred to Phase 8)**:
   - NGOs **do not** receive funds directly upon donor checkout.
   - Payouts to NGOs are strictly contingent on submitted evidence files (Phase 5), AI forensics verification (Phase 6), and human reviewer approval (Phase 7).
   - In Phase 4, zero disbursements occur. All captured funds remain safely in the platform's custody.
3. **No False Escrow Claims**:
   - CharityChain is transparent in its documentation: holding captured donor contributions in a platform merchant account does not constitute a legally chartered escrow trust under financial banking laws. The platform holds contributions as an intermediary subject to verified milestone progression.

---

## 3. Financial Reconciliation & Funding Aggregation

Project and milestone funding progress is calculated through database aggregation queries over settled transactions:

```python
# Net Funding Aggregation Formula
gross_captures = SUM(PaymentCapture.amount_minor_units WHERE status = 'COMPLETED')
completed_refunds = SUM(PaymentRefund.amount_minor_units WHERE status = 'COMPLETED')
net_funding = gross_captures - completed_refunds
```

The endpoint `GET /api/v1/projects/{id}/funding` returns:
```json
{
  "project_id": "94a257dc-0c7f-4d07-9431-6f4723b25a3b",
  "currency": "USD",
  "target_amount_minor_units": 1000000,
  "gross_received_minor_units": 100000,
  "refunded_minor_units": 20000,
  "net_received_minor_units": 80000,
  "progress_percentage": 8.0,
  "donations_count": 5,
  "milestones": [
    {
      "milestone_id": "afef3a8f-a0e3-41fb-8ef9-e93a856f3065",
      "sequence_number": 1,
      "title": "Site Survey",
      "target_amount_minor_units": 400000,
      "raised_amount_minor_units": 80000,
      "progress_percentage": 20.0,
      "status": "ACTIVE"
    }
  ]
}
```
