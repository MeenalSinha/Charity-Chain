# CharityChain — Analytics API Reference

## 1. Overview
The Analytics & Public Transparency API provides role-authorized analytics for NGOs, donors, and administrators, alongside unauthenticated endpoints for public inspection.

Base URL prefix: `/api/v1`

---

## 2. Public Transparency Endpoints (Unauthenticated)

### `GET /transparency/overview`
Retrieves platform-wide public transparency indicators.
- **Auth**: None
- **Response**: `200 OK`
```json
{
  "organizations": { "total": 12, "verified": 8 },
  "projects": { "total_published": 25, "active": 18, "completed": 7 },
  "milestones": { "total": 85, "completed": 54, "verified": 50 },
  "funding": {
    "total_captured_cents": 15000000,
    "total_refunded_cents": 250000,
    "net_funding_cents": 14750000,
    "total_disbursed_cents": 11000000,
    "total_captured_formatted": "$150,000.00",
    "net_funding_formatted": "$147,500.00",
    "total_disbursed_formatted": "$110,000.00"
  },
  "impact": { "certificates_issued": 142 },
  "as_of_date": "2026-11-13T16:00:00Z"
}
```

### `GET /transparency/projects`
Lists all public charitable projects.
- **Auth**: None
- **Parameters**: `limit` (int, default 50), `offset` (int, default 0)
- **Response**: `200 OK` (Array of public project summaries)

### `GET /transparency/projects/{project_id}`
Retrieves public project profile, milestones, and aggregated funding progress.
- **Auth**: None
- **Response**: `200 OK`

### `GET /transparency/organizations`
Lists verified non-profit organizations and their track records.
- **Auth**: None
- **Response**: `200 OK` (Array of verified NGO profiles)

### `GET /transparency/organizations/{organization_id}`
Retrieves public performance record for a specific NGO.
- **Auth**: None
- **Response**: `200 OK`

### `GET /transparency/geography`
Returns public map coordinates and country-level project density.
- **Auth**: None
- **Response**: `200 OK`
```json
{
  "points": [
    {
      "project_id": "proj-123",
      "title": "Clean Water Wells",
      "latitude": 4.12,
      "longitude": 11.51,
      "is_coarsened": true,
      "privacy_notice": "Location generalized to protect beneficiary security and operational safety."
    }
  ],
  "country_aggregates": [
    { "country_code": "KE", "project_count": 5, "milestones_completed": 12, "total_funding_cents": 2500000 }
  ]
}
```

### `GET /transparency/methodology`
Discloses calculation methodology and metrics formulas.
- **Auth**: None
- **Response**: `200 OK`

---

## 3. NGO Analytics Endpoints (Authenticated)

### `GET /analytics/ngo/{organization_id}/overview`
Returns high-level NGO KPIs (funding, milestones, disbursements, donor metrics).
- **Auth**: Bearer Token (`OWNER`, `ADMIN`, or `PLATFORM_ADMIN`)
- **Response**: `200 OK`

### `GET /analytics/ngo/{organization_id}/performance`
Returns project funding breakdown, donor retention rate, and donation size distribution.
- **Auth**: Bearer Token (`OWNER`, `ADMIN`, or `PLATFORM_ADMIN`)
- **Response**: `200 OK`

### `GET /analytics/ngo/{organization_id}/evidence-quality`
Returns evidence upload count, AI verification scores, and tampering incident metrics.
- **Auth**: Bearer Token (`OWNER`, `ADMIN`, or `PLATFORM_ADMIN`)
- **Response**: `200 OK`

### `GET /analytics/ngo/{organization_id}/report`
Generates an executive impact and performance report in plain text or Markdown.
- **Auth**: Bearer Token (`OWNER`, `ADMIN`, or `PLATFORM_ADMIN`)
- **Response**: `200 OK`

---

## 4. Donor Analytics Endpoints (Authenticated)

### `GET /analytics/donor/overview`
Returns personal giving KPIs (multi-currency totals, supported projects, completed milestones, certificates).
- **Auth**: Bearer Token (Authenticated Donor)
- **Response**: `200 OK`

### `GET /analytics/donor/portfolio`
Returns detailed giving breakdown by charitable category and organization.
- **Auth**: Bearer Token (Authenticated Donor)
- **Response**: `200 OK`

### `GET /analytics/donor/journey`
Returns the end-to-end contribution-to-impact journey for each donation.
- **Auth**: Bearer Token (Authenticated Donor)
- **Response**: `200 OK`

### `GET /analytics/donor/annual-report?year={year}`
Generates annual giving tax disclosure statement for a calendar year.
- **Auth**: Bearer Token (Authenticated Donor)
- **Response**: `200 OK`

---

## 5. Administrative Analytics Endpoints (Authenticated)

### `GET /analytics/admin/overview`
Global platform telemetry and overview.
- **Auth**: Bearer Token (`PLATFORM_ADMIN`)
- **Response**: `200 OK`

### `GET /analytics/admin/financial-operations`
Payment success rates, refund rates, payout performance, and ledger trial balance check.
- **Auth**: Bearer Token (`PLATFORM_ADMIN`, `FINANCE_AUDITOR`)
- **Response**: `200 OK`

### `GET /analytics/admin/review-workload`
Review case throughput, reviewer assignments, and verification backlog telemetry.
- **Auth**: Bearer Token (`PLATFORM_ADMIN`)
- **Response**: `200 OK`

---

## 6. Export Endpoints

### `GET /analytics/export/donations/csv`
Exports donations to CSV with formula injection sanitization.
- **Auth**: Bearer Token (`PLATFORM_ADMIN` or NGO `OWNER`/`ADMIN`)
- **Response**: `200 OK` (`text/csv`)
