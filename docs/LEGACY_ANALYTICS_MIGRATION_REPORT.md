# CharityChain — Legacy Analytics Migration Report
## Comprehensive Audit, Deprecation Plan & PostgreSQL Migration Strategy

- **Date:** 2026-11-13
- **Author:** CharityChain Architecture Team
- **Document Version:** 1.0.0
- **Migration Target:** Phase 11 (Analytics, Impact Reporting & Public Transparency)

---

## 1. Executive Summary

CharityChain's early prototypes (Phase 0) utilized an embedded **TinyDB** instance (`data/charity.db`) and simulated cryptocurrency metrics (`ETH`, 18-decimal minor units, and simulated NFT records) for prototyping dashboards, maps, and badges.

Since Phase 1, CharityChain transitioned to an enterprise-grade architecture backed by **PostgreSQL (SQLAlchemy)**, **PayPal Payment Processing**, **Double-Entry Financial Ledgers (Phase 8)**, **Cryptographic Proof-of-Impact Certificates (Phase 9)**, and **Transactional Event Outboxes (Phase 10)**.

However, several legacy analytics pathways in `charity_tracker.py` and `app_ui.py` remained tied to TinyDB records, calculating misleading metrics such as confusing milestones with projects, dividing fiat cents by $10^{18}$ as if they were Ethereum Wei, and ignoring payment captures, refunds, and double-entry reconciliation.

This report establishes the authoritative data sources, documents every legacy metric investigated, and outlines the non-breaking migration strategy for Phase 11.

---

## 2. Audit of Legacy Analytics Components

### 2.1 Authoritative Data Source Determination

| Domain Area | Legacy Implementation (TinyDB) | Authoritative Source (SQLAlchemy / PostgreSQL) | Status |
| :--- | :--- | :--- | :--- |
| **Organizations** | `db.ngo_profiles` (unauthenticated address key) | `backend.app.db.models.users.Organization` | **Migrated** |
| **Projects** | Confused with milestones | `backend.app.db.models.projects.Project` | **Migrated** |
| **Milestones** | `db.milestones` table | `backend.app.db.models.projects.Milestone` | **Migrated** |
| **Donations** | `db.donations` (unverified integer Wei) | `backend.app.db.models.financial.Donation` & `PaymentCapture` | **Migrated** |
| **Disbursements** | Not tracked | `backend.app.db.models.financial.Disbursement` & `JournalLine` | **Authoritative** |
| **Financial Ledger** | Absent | `backend.app.db.models.ledger.JournalEntry` (Double-entry) | **Authoritative** |
| **Certificates** | `db.nfts` (mock hash string) | `backend.app.db.models.certificates.Certificate` (RSA-PSS / Ed25519) | **Migrated** |
| **AI Verification** | `db.analytics` & `db.verifications` | `backend.app.db.models.verification.VerificationReport` | **Migrated** |
| **Geographic Data** | Raw milestone lat/long in TinyDB | `Project.latitude`, `Project.longitude`, `is_location_sensitive` | **Migrated** |

---

## 3. Discovered Legacy Calculation Defects

1. **Milestone vs. Project Conflation:**
   - *Legacy:* `len(milestones)` was reported as "Total Projects". In reality, projects contain multiple sequential milestones.
   - *Fix:* Dedicated SQL aggregations count distinct `Project.id` and distinct `Milestone.id` independently.

2. **Cryptocurrency Minor Unit Distortion:**
   - *Legacy:* Donations were divided by $10^{18}$ and labeled as `ETH` (`total_raised = sum(...) / 10**18 ETH`).
   - *Fix:* CharityChain uses standard fiat currency (USD, EUR, GBP) stored in **integer minor units (cents)**. All analytics aggregate cents and format as standard decimal currency (`$XX.XX`) with multi-currency separation.

3. **Omission of Refunds & Captures:**
   - *Legacy:* TinyDB `donations` were summed without checking payment status or deducting refunds.
   - *Fix:* Analytics exclusively aggregate payments where `status == DonationStatus.PAID` and `PaymentCapture.status == PaymentCaptureStatus.COMPLETED`, explicitly subtracting `PaymentRefund.amount_cents`.

4. **Absence of Role-Based Tenant Isolation:**
   - *Legacy:* TinyDB analytics provided no authentication or authorization filters, leaking platform totals to anyone viewing the dashboard.
   - *Fix:* Phase 11 introduces backend API and service boundaries enforcing strict tenant isolation for NGOs, personal isolation for donors, restricted visibility for admins, and privacy-filtered public transparency views.

5. **Lack of Location Privacy:**
   - *Legacy:* `create_project_map` plotted raw lat/long coordinates directly with no option to protect vulnerable beneficiary shelters or medical outposts.
   - *Fix:* `GeographicAnalyticsService` checks `is_location_sensitive` to aggregate sensitive project coordinates into coarse regional centroids, preventing exact address leakage.

---

## 4. Migration & Replacement Strategy

### 4.1 Retained & Reused Components
- **Visualization Helpers:** Plotly chart creation patterns in `charity_tracker.VisualizationManager` (e.g. gauge charts and confidence panels) are preserved for backward compatibility.
- **Folium & Plotly Map Integration:** Interactive map tile rendering is retained but powered by authoritative `Project` data with geographic clustering.
- **Badge Progression:** `BadgeSystem` is maintained in `charity_tracker.py` to ensure existing functional tests (`tests/functional/test_analytics_and_badges.py`) pass without modification.

### 4.2 Replaced Components
- **`show_analytics_dashboard()` in `app_ui.py`:** Replaced by a multi-tab, role-aware **Analytics & Transparency** workspace backed by `PlatformAnalyticsService`, `NGOAnalyticsService`, `DonorAnalyticsService`, and `PublicTransparencyService`.
- **`create_project_map()`:** Enhanced into an interactive, privacy-preserving geographic impact map that queries published `Project` records from the database.
- **`page_home()` KPI Cards:** Updated to display authoritative fiat metrics (`Total Captured Funding`, `Active Projects`, `Completed Milestones`, `Issued Certificates`) instead of TinyDB ETH calculations.

---

## 5. Backward Compatibility & Test Suite Invariance

To guarantee that no regressions occur:
- `charity_tracker.py` maintains its existing class interfaces (`DatabaseManager`, `VisualizationManager`, `BadgeSystem`) so existing unit tests continue to pass without error.
- All newly authored analytics services operate within `backend/app/services/analytics/` with dedicated SQLAlchemy session management.
- The Alembic migration `011_phase11_analytics_and_transparency.py` safely introduces analytical snapshot tables without altering or dropping existing financial, certificate, or notification tables.

