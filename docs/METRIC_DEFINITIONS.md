# CharityChain — Central Metric Definitions & Taxonomy
## Authoritative Formula Catalog, Data Lineage & Calculation Rules

- **Document Version:** 1.0.0
- **Phase:** Phase 11 (Analytics, Impact Reporting & Public Transparency)
- **Authoritative Ledger:** PostgreSQL & SQLAlchemy Models
- **Standard Currency Representation:** Integer Minor Units (Cents)

---

## 1. Overview & Guiding Philosophy

CharityChain enforces that **every published metric must have a defined calculation, a trustworthy source, and a clear interpretation.** Fabricated, hardcoded, or ambiguous statistics are strictly prohibited across all public, NGO, donor, and administrative views.

All monetary values are calculated in **integer minor units (cents)** with currency-level segregation, preventing floating-point approximation inaccuracies and inappropriate cross-currency summing.

---

## 2. Central Metric Catalog

| Metric ID | Human-Readable Name | Data Source | Calculation Formula | Included Statuses | Excluded Statuses | Privacy Tier | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `total_captured_donations` | Total Captured Donations | `payment_captures` | $\sum(\text{PaymentCapture.amount\_minor\_units})$ | `COMPLETED` | `PENDING`, `FAILED`, `VOIDED` | `PUBLIC` | cents |
| `net_donations_after_refunds` | Net Donations After Refunds | `payment_captures`, `payment_refunds` | $\text{total\_captured} - \sum(\text{PaymentRefund.amount\_minor\_units})$ | `COMPLETED` | `PENDING`, `FAILED` | `PUBLIC` | cents |
| `confirmed_disbursements` | Confirmed Disbursements | `disbursements` | $\sum(\text{Disbursement.amount\_minor\_units})$ | `COMPLETED` | `REQUESTED`, `APPROVED`, `SUBMITTED`, `FAILED` | `PUBLIC` | cents |
| `active_projects` | Active Projects | `projects` | $\text{COUNT}(\text{DISTINCT Project.id})$ | `ACTIVE` | `DRAFT`, `PENDING_REVIEW`, `COMPLETED` | `PUBLIC` | count |
| `completed_projects` | Completed Projects | `projects` | $\text{COUNT}(\text{DISTINCT Project.id})$ | `COMPLETED` | `DRAFT`, `ACTIVE`, `SUSPENDED` | `PUBLIC` | count |
| `total_milestones` | Total Milestones | `milestones` | $\text{COUNT}(\text{DISTINCT Milestone.id})$ | `ALL` | None | `PUBLIC` | count |
| `completed_milestones` | Completed Milestones | `milestones` | $\text{COUNT}(\text{DISTINCT Milestone.id})$ | `COMPLETED` | `DRAFT`, `PENDING`, `ACTIVE`, `SUBMITTED` | `PUBLIC` | count |
| `verified_milestones` | Verified Milestones | `milestones`, `review_cases` | $\text{COUNT}(\text{DISTINCT Milestone.id})$ | `APPROVED`, `COMPLETED` | `DRAFT`, `PENDING`, `REJECTED` | `PUBLIC` | count |
| `evidence_review_completion_rate` | Review Completion Rate | `review_cases` | $(\frac{\text{approved} + \text{rejected}}{\text{total\_cases}}) \times 100$ | `APPROVED`, `REJECTED` | `PENDING`, `UNDER_REVIEW`, `REVISION_REQUESTED` | `ADMIN_ONLY` | % |
| `donor_retention_rate` | Donor Retention Rate | `donations` | $(\frac{\text{donors with } > 1 \text{ donation}}{\text{total unique donors}}) \times 100$ | `COMPLETED` | `CREATED`, `FAILED`, `CANCELLED` | `NGO_RESTRICTED` | % |
| `ngo_project_completion_rate` | NGO Project Completion Rate | `projects` | $(\frac{\text{completed projects}}{\text{total active + completed}}) \times 100$ | `COMPLETED` | `DRAFT` | `PUBLIC` | % |
| `total_certificates_issued` | Proof-of-Impact Certificates Issued | `impact_certificates` | $\text{COUNT}(\text{DISTINCT ImpactCertificate.id})$ | `ACTIVE`, `ISSUED` | `REVOKED`, `SUSPENDED` | `PUBLIC` | count |
| `reconciliation_discrepancies_count`| Open Settlement Discrepancies | `reconciliation_items` | $\text{COUNT}(\text{ReconciliationItem.id})$ | `DISCREPANCY` (unresolved) | `MATCHED`, `RESOLVED` | `ADMIN_ONLY` | count |

---

## 3. Financial Consistency & Ledger Integrity Rules

1. **Pre-Authorization Holds Excluded:**
   - Card authorization holds (`PaymentAuthorizationStatus.AUTHORIZED`) represent temporary guarantees, not settled platform funds. They are strictly excluded from gross and net donation metrics.
2. **Refund Deductions:**
   - When a donor refund or reversal is finalized (`PaymentRefundStatus.COMPLETED`), it is subtracted from net platform funding and the project's net raised balance.
3. **Double-Entry Ledger Balancing:**
   - Operational health metrics verify that the sum of debit journal lines equals the sum of credit journal lines across all ledger accounts:
     $$\sum \text{Debit Lines} = \sum \text{Credit Lines}$$
   - Any non-zero difference triggers an administrative warning banner in the Administrator Health tab.

