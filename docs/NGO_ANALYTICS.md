# CharityChain — NGO Analytics & Impact Performance

## 1. Overview & Purpose
Phase 11 introduces a multi-dimensional, tenant-isolated analytics suite specifically built for Non-Governmental Organizations (NGOs). Prior to Phase 11, project managers relied on static counters or unverified database entries. With the Phase 11 NGO Analytics service, organization administrators and project managers have immediate access to real-time, authoritative metrics derived from:
- **Direct Donor Contributions**: Segregated by currency and status (`DonationStatus.PAID`).
- **Payment Captures & Refunds**: PayPal-verified gross captures minus processed refunds.
- **Milestone Progress Tracking**: Completion and verification rates across active and archived projects.
- **Evidence & Verification Quality**: Distribution of AI confidence scores, document tampering flags, and reviewer outcomes.
- **Donor Retention & Size Distribution**: Cohort-based repeat donation rates and tier distribution.

---

## 2. Key Metrics & Formulas

### 2.1 Funding Overview
| Metric | Formula | Source Table | Description |
| :--- | :--- | :--- | :--- |
| **Gross Captured Funding** | `SUM(Donation.amount_minor_units)` where `status == PAID` and project belongs to NGO | `donations`, `projects` | Total donations captured from donors for the NGO's projects. |
| **Processed Refunds** | `SUM(PaymentRefund.amount_minor_units)` where `status == COMPLETED` | `payment_refunds`, `payment_captures`, `donations` | Total funds returned to donors due to cancellations or disputes. |
| **Net Funding** | `MAX(0, Gross Captured - Processed Refunds)` | Derived | Authoritative net funds available for project milestones. |
| **Confirmed Disbursements** | `SUM(Disbursement.amount_minor_units)` where `status == SUCCEEDED` | `disbursements` | Total payouts transferred to NGO bank/PayPal accounts upon milestone approval. |
| **Undisbursed / Available Funds** | `MAX(0, Net Funding - Confirmed Disbursements)` | Derived | Balance currently held in platform escrow pending milestone milestones. |

### 2.2 Project & Milestone Lifecycle KPIs
- **Total Projects**: `COUNT(Project.id)` scoped to the organization.
- **Active Projects**: Projects with `status == ACTIVE`.
- **Completed Projects**: Projects with `status == COMPLETED`.
- **Milestone Completion Rate**: `(Completed Milestones / Total Milestones) * 100`.
- **Milestone Verification Rate**: `(Verified Milestones / Total Milestones) * 100`, where verified includes `APPROVED` and `COMPLETED`.

### 2.3 Donor Engagement & Retention
- **Total Unique Donors**: `COUNT(DISTINCT Donation.donor_id)` where `donor_id IS NOT NULL`.
- **Repeat Donors**: Count of unique donor IDs having `COUNT(Donation.id) > 1`.
- **Donor Retention Rate**: `(Repeat Donors / Total Unique Donors) * 100`.
- **Donation Size Distribution**: Bucketed into standard tiers:
  - Small: `< $25.00` (< 2,500 cents)
  - Medium: `$25.00 – $99.99` (2,500 – 9,999 cents)
  - Large: `$100.00 – $499.99` (10,000 – 49,999 cents)
  - Major: `≥ $500.00` (≥ 50,000 cents)

---

## 3. Evidence Quality & AI Verification Diagnostics
NGO performance is evaluated based on the cryptographic and forensic fidelity of uploaded evidence files:
- **Total Evidence Files Uploaded**: Scoped to milestones under the NGO's projects.
- **Verified Evidence Files**: Files with `is_verified == True`.
- **Average AI Confidence Score**: Mean indicator confidence across all completed `VerificationReport` records for the organization's milestones.
- **Tampering & Anomaly Incidents**: Count of reports with `tampering_detected == True` or confidence score < 0.40.

---

## 4. Tenant Isolation & Authorization
All NGO analytics endpoints enforce role-based access control (RBAC) and strict organization scoping:
- Only organization members with roles `OWNER` or `ADMIN`, or users with platform role `PLATFORM_ADMIN`, can access `/api/v1/analytics/ngo/{organization_id}/*`.
- SQL queries explicitly filter by `Project.organization_id == organization_id`.
- Requests from unauthorized users or members of other organizations are rejected with HTTP 403 Forbidden.

---

## 5. UI Integration
In the Streamlit application (`app_ui.py`):
1. **NGO Analytics Tab**: Located in the main navigation under `Analytics & Transparency`.
2. **KPI Cards**: Displays net raised, total disbursements, available balance, and milestone completion.
3. **Funding Breakdown Chart**: Interactive bar/progress breakdown comparing target vs. raised funds per project.
4. **Donor Tier Breakdown**: Visual breakdown of donor donation sizes and retention percentages.
5. **One-Click Impact Report Generator**: Generates and downloads an exhaustive Markdown or CSV performance report for the organization.
