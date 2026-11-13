# CharityChain — Public Transparency & Open Governance

## 1. Overview & Architectural Principles
Public trust in charitable giving requires unassailable transparency without compromising donor privacy or beneficiary safety. Phase 11 implements a dedicated Public Transparency service (`PublicTransparencyService`) accessible to unauthenticated visitors, researchers, regulators, and the press.

### Core Tenets
1. **Authoritative Ledger & Payment Records**: Metrics are computed exclusively from PostgreSQL database state (PayPal captures, refunds, disbursements, and double-entry ledger lines). Legacy unverified datastores (such as TinyDB) have been completely deprecated.
2. **Zero Personally Identifiable Information (PII)**: Public endpoints never emit individual donor names, email addresses, IP addresses, or internal payment gateway order tokens.
3. **Location Privacy**: GPS coordinates for sensitive projects (such as domestic violence shelters, protected wildlife reserves, or frontline humanitarian medical posts) are coarsened to regional centroids (rounded to 1 decimal place, ~11.1 km resolution).
4. **Methodological Disclosure**: All public KPI calculations are published alongside explicit mathematical definitions and data source disclosures.

---

## 2. Public Transparency Endpoints

### 2.1 Global Platform Overview (`GET /api/v1/transparency/overview`)
Returns global charitable metrics:
- **Organizations**: Total registered NGOs and total verified NGOs.
- **Projects**: Total published public projects, active projects, and successfully completed projects.
- **Milestones**: Total published milestones, completed milestones, and dual-key verified milestones.
- **Financial Aggregates**:
  - `total_captured_cents`: Total gross donations confirmed through payment providers.
  - `total_refunded_cents`: Total refunded amounts.
  - `net_funding_cents`: Authoritative net funds raised (`captured - refunded`).
  - `total_disbursed_cents`: Total funds disbursed to NGOs upon milestone completion.
  - `impact_certificates_issued`: Total active Proof-of-Impact certificates issued.

### 2.2 Public Project Directory (`GET /api/v1/transparency/projects`)
Returns public projects with visibility `ProjectVisibility.PUBLIC`:
- Excludes draft, internal, or unapproved projects.
- Includes target funding, raised funding, progress percentages, milestone counts, and verified NGO attribution.

### 2.3 Verified NGO Directory (`GET /api/v1/transparency/organizations`)
Provides transparency profiles for verified non-profit partners:
- Organization legal name, verification status, and active project roster.
- Financial overview: Total funds raised and total disbursements received.
- Milestone completion track record and aggregate performance.

### 2.4 Calculation Methodology (`GET /api/v1/transparency/methodology`)
Discloses the technical formulas, reconciliation processes, and audit mechanisms backing all published metrics.

---

## 3. Location-Sensitive Safety Guardrails
When a project or milestone has `is_location_sensitive = True`:
- Exact GPS coordinates (`latitude`, `longitude`) are stripped from public responses.
- The platform substitutes a regional centroid coordinate (`round(lat, 1)`, `round(lon, 1)`).
- The marker is labeled with `is_coarsened: true` and includes the explanation: *"Location generalized to protect beneficiary security and operational safety."*

---

## 4. UI Transparency Portal
The Streamlit interface (`app_ui.py`) includes a dedicated **Public Transparency** page (`page_public_transparency()`):
- **Live Platform Metrics**: Top-level metric cards with real-time updates.
- **Project Transparency Explorer**: Searchable and filterable list of public initiatives.
- **Verified Organizations Directory**: Track record and impact achievements of NGO partners.
- **Calculation Methodology & Disclosure**: Direct link to formulas and open audit standards.
