# CharityChain — API Design Specification
**Phase 1 Deliverable — Versioned REST API v1, Standard Envelopes & Error Contracts**

---

## 1. Architectural & Protocol Conventions

The CharityChain API is designed as a versioned, stateless, predictable RESTful service following OpenAPI 3.1 standards:
- **Base URI**: `http://localhost:8000/api/v1`
- **Transport**: HTTPS (HTTP in local development)
- **Data Exchange**: JSON (`application/json`) with UTF-8 encoding
- **Idempotency**: All mutating financial and disbursement endpoints require an `Idempotency-Key: <UUID>` header.
- **Correlation ID**: Every request/response cycle includes `X-Correlation-ID: <UUID>` for cross-tier distributed tracing.

---

## 2. Standard Response Envelopes

Every endpoint response adheres to consistent envelope structures:

### 2.1 Standard Success Envelope
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully.",
  "timestamp": "2026-11-11T12:00:00Z",
  "correlation_id": "c1f7b8a2-9e23-4d7a-85b4-3a9d9c72e1a4"
}
```

### 2.2 Standard Paginated Envelope
```json
{
  "success": true,
  "data": [ ... ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 142,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  },
  "timestamp": "2026-11-11T12:00:00Z",
  "correlation_id": "c1f7b8a2-9e23-4d7a-85b4-3a9d9c72e1a4"
}
```

### 2.3 Standard Error Envelope
```json
{
  "success": false,
  "error": {
    "code": "INVALID_STATE_TRANSITION",
    "message": "Milestone cannot transition from NOT_STARTED to VERIFIED directly.",
    "details": {
      "current_status": "NOT_STARTED",
      "target_status": "VERIFIED",
      "allowed_transitions": ["IN_PROGRESS", "CANCELLED"]
    }
  },
  "timestamp": "2026-11-11T12:00:00Z",
  "correlation_id": "c1f7b8a2-9e23-4d7a-85b4-3a9d9c72e1a4"
}
```

---

## 3. Standardized Error Codes & HTTP Mappings

| Error Code | HTTP Status | Description |
|---|---|---|
| `VALIDATION_ERROR` | 422 Unprocessable Entity | Schema, field type, or business constraint failure |
| `UNAUTHORIZED` | 401 Unauthorized | Missing or expired authentication token |
| `FORBIDDEN` | 403 Forbidden | User role lacks permission for this action |
| `NOT_FOUND` | 404 Not Found | Requested entity UUID does not exist |
| `CONFLICT` | 409 Conflict | Duplicate resource (e.g. duplicate email, slug) |
| `INVALID_STATE_TRANSITION` | 400 Bad Request | Action violates domain state machine rules |
| `FILE_TOO_LARGE` | 413 Payload Too Large | Uploaded evidence exceeds 10 MB limit |
| `INVALID_FILE_TYPE` | 415 Unsupported Media Type | File magic bytes do not match allowed image MIME types |
| `RATE_LIMIT_EXCEEDED` | 429 Too Many Requests | Upload or request rate exceeds hourly threshold |
| `SERVICE_UNAVAILABLE` | 503 Service Unavailable | External provider (PayPal, AI engine) offline |
| `INTERNAL_SERVER_ERROR` | 500 Internal Server Error | Unexpected server error (logged with traceback) |

---

## 4. Complete Endpoint Catalog

### 4.1 System & Observability
- `GET /api/v1/health`
  - Returns service status, version, uptime, and runtime environment.
- `GET /api/v1/health/db`
  - Verifies live connectivity to PostgreSQL database pool.

### 4.2 Projects & Milestones
- `GET /api/v1/projects`
  - Query parameters: `page`, `page_size`, `category`, `status`, `organization_id`
  - Returns paginated list of projects.
- `POST /api/v1/projects`
  - Body: `{ organization_id, title, description, target_amount_minor, currency, category_id }`
  - Creates a new project in `DRAFT` status.
- `GET /api/v1/projects/{id}`
  - Returns single project with milestone summaries and funding statistics.
- `GET /api/v1/projects/{id}/milestones`
  - Returns ordered list of milestones for project.
- `POST /api/v1/projects/{id}/milestones`
  - Body: `{ sequence_order, title, description, target_amount_minor, due_date }`
  - Adds a milestone to a project.
- `GET /api/v1/milestones/{id}`
  - Returns milestone details, evidence records, and verification status.

### 4.3 Donations & Payment Workflows
- `GET /api/v1/donations`
  - Returns user's donation history.
- `POST /api/v1/donations`
  - Body: `{ project_id, amount_minor, currency, is_anonymous }`
  - Creates a pending donation record.
- `POST /api/v1/payments/create-order`
  - Body: `{ donation_id }`
  - Interacts with PayPal to initialize order with `intent: AUTHORIZE`.
  - Returns PayPal approve URL and `gateway_order_id`.
- `POST /api/v1/payments/authorize`
  - Body: `{ order_id }`
  - Authorizes funds upon donor approval; funds held in escrow.
- `POST /api/v1/payments/capture`
  - Body: `{ authorization_id }`
  - Internal or authorized reviewer trigger upon verified milestone completion.
- `POST /api/v1/disbursements/execute`
  - Body: `{ milestone_id }`
  - Initiates payment to NGO PayPal account for completed milestone.

### 4.4 Evidence & Verification
- `POST /api/v1/milestones/{id}/evidence`
  - Multipart Form: `file` (binary), `uploader_id`
  - Enforces magic bytes check, file size limit (10MB), and calculates SHA-256 and pHash.
- `GET /api/v1/evidence/{id}`
  - Returns evidence metadata and download URI.
- `GET /api/v1/evidence/{id}/verification`
  - Returns automated AI verification report and forensic indicators (YOLO, ELA, SSIM, GPS).

### 4.5 Human Review & Governance
- `GET /api/v1/reviews/tasks`
  - Query parameters: `status`, `assigned_to`
  - Returns queue of pending evidence review tasks.
- `POST /api/v1/reviews/tasks/{id}/decide`
  - Body: `{ decision: "APPROVED" | "REJECTED" | "MORE_INFO_REQUESTED", notes }`
  - Records human decision; transitions milestone status accordingly.

### 4.6 Impact Certificates & Ledger
- `GET /api/v1/certificates`
  - Returns issued Proof-of-Impact certificates.
- `GET /api/v1/certificates/{id}/verify`
  - Public verification endpoint validating certificate SHA-256 digest against Impact Ledger.

