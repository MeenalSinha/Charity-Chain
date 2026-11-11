# CharityChain — Organization, Project & Milestone REST API Specification

## 1. Overview & Authentication

All API endpoints are mounted under `/api/v1`. 

### Headers
| Header | Description | Requirement |
|---|---|---|
| `Authorization` | `Bearer <access_token>` | Required for all authenticated NGO, Reviewer, and Admin endpoints |
| `Content-Type` | `application/json` or `multipart/form-data` | Required as specified per endpoint |

### Standard Response Envelope
All non-binary endpoints return a standard envelope:
```json
{
  "success": true,
  "data": { ... },
  "message": null
}
```

Paginated endpoints return:
```json
{
  "success": true,
  "data": [ ... ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 42,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

### Standard Error Envelope
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable error explanation",
    "details": { ... }
  }
}
```

---

## 2. Organization Endpoints

### 2.1 Create Organization
- **Endpoint**: `POST /api/v1/organizations`
- **Auth**: Authenticated User
- **Request Body**:
  ```json
  {
    "legal_name": "Hope Water Initiative",
    "registration_number": "NGO-2026-9921",
    "tax_id": "TAX-US-88291",
    "country_of_registration": "USA",
    "address": "123 Charity Way, Seattle, WA",
    "website_url": "https://hopewater.org",
    "mission_statement": "Providing clean drinking water to remote communities."
  }
  ```
- **Response**: `201 Created`
  ```json
  {
    "success": true,
    "data": {
      "id": "c1f7b02e-99f5-46aa-a185-3e2849bca722",
      "legal_name": "Hope Water Initiative",
      "status": "DRAFT",
      "verification_level": "NONE",
      "country_of_registration": "USA",
      "created_at": "2026-11-11T12:00:00Z"
    }
  }
  ```

### 2.2 Get Organization Details
- **Endpoint**: `GET /api/v1/organizations/{organization_id}`
- **Auth**: Organization Member, Reviewer, or Platform Admin
- **Response**: `200 OK`

### 2.3 Update Organization
- **Endpoint**: `PATCH /api/v1/organizations/{organization_id}`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Request Body**:
  ```json
  {
    "address": "456 New Mission Rd, Seattle, WA",
    "website_url": "https://new.hopewater.org"
  }
  ```
- **Response**: `200 OK`

### 2.4 Upload Verification Document
- **Endpoint**: `POST /api/v1/organizations/{organization_id}/documents`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `document_type`: `REGISTRATION_CERTIFICATE` | `TAX_EXEMPTION` | `ANNUAL_REPORT` | `FINANCIAL_AUDIT` | `BANK_STATEMENT`
  - `title`: Document display title
  - `file`: Binary file (PDF, PNG, or JPEG, $\le 10\text{MB}$)
- **Response**: `201 Created`
  ```json
  {
    "success": true,
    "data": {
      "id": "e30129ad-...",
      "document_type": "REGISTRATION_CERTIFICATE",
      "file_name": "cert_2026.pdf",
      "mime_type": "application/pdf",
      "file_size": 204850,
      "checksum_sha256": "4a71b...8b2",
      "uploaded_at": "2026-11-11T12:05:00Z"
    }
  }
  ```

### 2.5 Submit Organization for Verification
- **Endpoint**: `POST /api/v1/organizations/{organization_id}/verification-submission`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Validation**: Requires `DRAFT` status and at least one uploaded verification document.
- **Response**: `200 OK` (Status transitions to `PENDING`)

---

## 3. Project Endpoints

### 3.1 Discover Public Projects
- **Endpoint**: `GET /api/v1/projects`
- **Auth**: Public (No auth required)
- **Query Parameters**:
  - `search`: Keyword string (matches title or summary)
  - `category`: Category slug
  - `country`: Country ISO code
  - `sort`: `created_desc`, `target_asc`, `target_desc`
  - `page`: Page index (default: `1`)
  - `page_size`: Results per page (default: `20`, max: `100`)
- **Security Invariant**: Returns only projects where `status == ACTIVE`, `public_visibility == PUBLIC`, and sponsoring organization `status == VERIFIED`.
- **Response**: `200 OK` (Paginated list of `ProjectResponse`)

### 3.2 Create Project
- **Endpoint**: `POST /api/v1/projects`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Request Body**:
  ```json
  {
    "organization_id": "c1f7b02e-99f5-46aa-a185-3e2849bca722",
    "title": "Clean Water Well Construction",
    "summary": "Drilling and installing 3 solar-powered water boreholes.",
    "objectives": "Provide potable water to 1,500 villagers.",
    "project_type": "INFRASTRUCTURE",
    "category": "clean-water",
    "funding_target_minor": 1500000,
    "currency": "USD",
    "country": "KEN",
    "location": "Turkana County",
    "public_visibility": "PUBLIC"
  }
  ```
- **Response**: `201 Created`

### 3.3 Get Project Details
- **Endpoint**: `GET /api/v1/projects/{project_id}`
- **Auth**: Public if active, else Organization Member, Reviewer, or Platform Admin
- **Response**: `200 OK`

### 3.4 Update Project
- **Endpoint**: `PATCH /api/v1/projects/{project_id}`
- **Auth**: Organization `OWNER` or `ADMIN` (Project must be in `DRAFT` or `REVISION_REQUIRED` status)
- **Response**: `200 OK`

### 3.5 Run Pre-Submission Live Validation
- **Endpoint**: `GET /api/v1/projects/{project_id}/validation`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Description**: Executes the 13-rule checklist without modifying status or raising errors.
- **Response**: `200 OK`
  ```json
  {
    "success": true,
    "data": {
      "valid": false,
      "total_issues": 1,
      "issues": [
        {
          "field": "milestones",
          "code": "MILESTONE_BUDGET_MISMATCH",
          "message": "Milestone funding sum (10000.00 USD) does not match project funding target (15000.00 USD)."
        }
      ]
    }
  }
  ```

### 3.6 Submit Project for Review
- **Endpoint**: `POST /api/v1/projects/{project_id}/submit`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Validation**: Strict execution of the 13-rule checklist. Fails with `422 Unprocessable Entity` if any rule fails.
- **Response**: `200 OK` (Status transitions to `SUBMITTED`)

### 3.7 Lifecycle Actions
- `POST /api/v1/projects/{project_id}/withdraw` — Withdraws submitted project to `DRAFT`.
- `POST /api/v1/projects/{project_id}/activate` — Activates `APPROVED` project for public display.
- `POST /api/v1/projects/{project_id}/pause` — Temporarily halts an `ACTIVE` project.
- `POST /api/v1/projects/{project_id}/resume` — Resumes a `PAUSED` project back to `ACTIVE`.
- `POST /api/v1/projects/{project_id}/complete` — Marks an `ACTIVE` project as completed.
- `POST /api/v1/projects/{project_id}/archive` — Platform Admin long-term archive.

---

## 4. Milestone Endpoints

### 4.1 Add Milestone
- **Endpoint**: `POST /api/v1/projects/{project_id}/milestones`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Request Body**:
  ```json
  {
    "sequence_number": 1,
    "title": "Geological Survey & Site Permitting",
    "objective": "Complete hydrological sounding and obtain local permits.",
    "deliverables": "Hydrology report and county government permit.",
    "expected_outcome": "Certified drill location.",
    "success_criteria": "Water table verified at >= 60 meters.",
    "funding_target_minor": 500000,
    "target_latitude": 3.1234,
    "target_longitude": 35.5987
  }
  ```
- **Validation**: Enforces non-excess budget allocation and sequential ordering.
- **Response**: `201 Created`

### 4.2 List Project Milestones
- **Endpoint**: `GET /api/v1/projects/{project_id}/milestones`
- **Auth**: Public if project is active, otherwise Organization Member or Reviewer.
- **Response**: `200 OK` (Ordered list of milestones by sequence number)

### 4.3 Update Milestone
- **Endpoint**: `PATCH /api/v1/milestones/{milestone_id}`
- **Auth**: Organization `OWNER` or `ADMIN`
- **Response**: `200 OK`

### 4.4 Delete Milestone
- **Endpoint**: `DELETE /api/v1/milestones/{milestone_id}`
- **Auth**: Organization `OWNER` or `ADMIN` (Project must be in `DRAFT` status)
- **Response**: `200 OK`

---

## 5. Reviewer & Admin Endpoints

### 5.1 Organization Review Queue
- **Endpoint**: `GET /api/v1/admin/organizations/queue`
- **Auth**: Platform `REVIEWER` or `PLATFORM_ADMIN`
- **Response**: `200 OK` (List of organizations in `PENDING` status)

### 5.2 Submit Organization Review Decision
- **Endpoint**: `POST /api/v1/admin/organizations/{organization_id}/decision`
- **Auth**: Platform `REVIEWER` or `PLATFORM_ADMIN`
- **Conflict of Interest**: Reviewer must not have any membership or role in the target organization. Returns `403 Forbidden` if conflict detected.
- **Request Body**:
  ```json
  {
    "decision": "APPROVED",
    "review_notes": "Official government accreditation documents and tax records verified."
  }
  ```
- **Valid Decisions**: `APPROVED` | `REJECTED` | `REVISIONS_REQUESTED`
- **Response**: `200 OK`

### 5.3 Project Review Queue
- **Endpoint**: `GET /api/v1/admin/projects/queue`
- **Auth**: Platform `REVIEWER` or `PLATFORM_ADMIN`
- **Response**: `200 OK` (List of projects in `SUBMITTED` status)

### 5.4 Submit Project Review Decision
- **Endpoint**: `POST /api/v1/admin/projects/{project_id}/decision`
- **Auth**: Platform `REVIEWER` or `PLATFORM_ADMIN`
- **Conflict of Interest**: Reviewer must not be affiliated with the sponsoring organization. Returns `403 Forbidden` if conflict detected.
- **Request Body**:
  ```json
  {
    "decision": "APPROVED",
    "review_notes": "Milestone deliverables, budget breakdowns, and coordinates confirmed."
  }
  ```
- **Valid Decisions**: `APPROVED` | `REJECTED` | `REVISIONS_REQUESTED`
- **Response**: `200 OK`
