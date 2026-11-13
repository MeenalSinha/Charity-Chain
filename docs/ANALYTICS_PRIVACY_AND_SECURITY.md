# CharityChain — Analytics Privacy & Security Architecture

## 1. Threat Model & Security Objectives
The Analytics & Public Transparency subsystem is designed to meet strict security and privacy standards:
1. **Tenant Isolation**: Prevent unauthorized access or data leakage between competing or separate NGOs.
2. **Donor Privacy & Anti-De-anonymization**: Protect individual donor identities, personal giving totals, and transaction histories from public exposure or cross-donor harvesting.
3. **Beneficiary Operational Security**: Safeguard physical locations of vulnerable beneficiaries and assets.
4. **Export Safety**: Mitigate formula injection (CSV injection) vulnerabilities in downloaded spreadsheet reports.

---

## 2. Tenant Isolation Enforcement

### 2.1 Role-Based Access Control (RBAC)
- **NGO Analytics Endpoints** (`/api/v1/analytics/ngo/{organization_id}/*`):
  - Requesting user must be authenticated.
  - User must hold membership in the specified organization with role `OWNER` or `ADMIN`, or possess platform-level role `PLATFORM_ADMIN`.
  - Non-members or members of different organizations attempting to query the organization's private analytics receive HTTP 403 Forbidden.
- **Donor Endpoints** (`/api/v1/analytics/donor/*`):
  - Automatically scoped to `current_user.id`.
  - Donors cannot query another user's donor portfolio.
- **Platform Operations Endpoints** (`/api/v1/analytics/admin/*`):
  - Restricted strictly to `PLATFORM_ADMIN`, `FINANCE_AUDITOR`, or `FINANCE_APPROVER`.

---

## 3. Donor Privacy & Anonymous Giving
- **Anonymous Flag**: When `Donation.is_anonymous == True`:
  - `donor_name` is suppressed in all public queries.
  - Endpoints return `"Anonymous Donor"` as the display name.
  - Email addresses, user IDs, and IP addresses are completely excluded from the projection.
- **Aggregation Thresholds**: Donor tier distributions group donations into standard brackets without disclosing exact timestamps or individual transaction links.

---

## 4. Location Privacy Protocols
- If `Project.is_location_sensitive == True` or `Milestone.is_location_sensitive == True`:
  - Precise coordinates from GPS hardware or evidence EXIF tags are masked.
  - The API exposes only the regional centroid (coordinates rounded to 1 decimal place, $\approx 11.1$ km radius).
  - Explicit warning flag `is_coarsened: true` is included with explanatory metadata.

---

## 5. CSV Formula Injection (CSV Injection) Mitigation
Exported CSV files can be opened by users in spreadsheet applications (Microsoft Excel, LibreOffice Calc, Google Sheets). If an attacker injects malicious spreadsheet formulas into project titles, donor names, or notes (e.g. `=cmd|' /C calc'!A0` or `-2+3+cmd...`), the spreadsheet application might execute arbitrary commands.

### Defense Mechanism
The `AnalyticsExportService` scans every cell before serializing to CSV:
- If a string cell starts with any formula trigger character:
  - `=` (Equals)
  - `+` (Plus)
  - `-` (Minus)
  - `@` (At sign)
  - `\t` (Tab)
  - `\r` (Carriage return)
- The cell content is prepended with a single quote (`'`), which instructs spreadsheet parsers to treat the value strictly as plain text.
- Tested and verified in `tests/unit/test_analytics_exports.py`.
