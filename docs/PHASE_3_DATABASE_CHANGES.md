# CharityChain — Phase 3 Database Schema & Migration Specification

## 1. Overview

Phase 3 introduces substantial database schema expansions to transition CharityChain from an authentication-focused baseline to a multi-tenant NGO onboarding and milestone-based project management platform.

All schema modifications are governed by Alembic migration `003_phase3_org_and_projects.py` (down revision: `002_phase2_identity_and_auth`) and are fully compatible with both PostgreSQL and SQLite.

---

## 2. Table Modifications & New Tables

### 2.1 Table: `organizations` (Altered)
The `organizations` table was expanded with legal compliance, registration metadata, and verification lifecycle fields:

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `legal_name` | VARCHAR(255) | NO | - | Formal legal name of the organization |
| `registration_authority` | VARCHAR(255) | YES | NULL | Government ministry or registrar of societies |
| `country` | VARCHAR(3) | YES | NULL | ISO 3166-1 alpha-3 country code |
| `registered_address` | TEXT | YES | NULL | Full physical registered legal address |
| `mission_statement` | TEXT | YES | NULL | Formal mission and charter summary |
| `focus_areas` | JSON / Text | YES | NULL | Structured list of charitable sectors (e.g. `["Water", "Education"]`) |
| `verification_requested_at`| TIMESTAMP | YES | NULL | Timestamp when submitted for review |
| `verified_at` | TIMESTAMP | YES | NULL | Timestamp of reviewer approval |
| `reviewer_id` | VARCHAR(36) | YES | NULL | Foreign key referencing `users.id` of the reviewing officer |
| `review_notes` | TEXT | YES | NULL | Notes or justifications provided during review |

---

### 2.2 Table: `organization_documents` (New)
Created to securely store uploaded NGO verification credentials (tax certificates, registration filings, financial audits):

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | PRIMARY KEY | UUIDv4 identifier |
| `organization_id` | VARCHAR(36) | FOREIGN KEY -> `organizations.id` | Owning organization |
| `document_type` | VARCHAR(50) | NOT NULL | Type enum (`REGISTRATION_CERTIFICATE`, `TAX_EXEMPTION`, etc.) |
| `file_name` | VARCHAR(255) | NOT NULL | Sanitized original filename |
| `file_path` | VARCHAR(512) | NOT NULL | Storage provider relative path |
| `mime_type` | VARCHAR(100) | NOT NULL | Verified MIME type (`application/pdf`, `image/png`, `image/jpeg`) |
| `file_size` | INTEGER | NOT NULL | File size in bytes ($\le 10\text{MB}$) |
| `checksum_sha256` | VARCHAR(64) | NOT NULL | Cryptographic SHA-256 hash |
| `uploaded_by_id` | VARCHAR(36) | FOREIGN KEY -> `users.id` | Uploader user account |
| `uploaded_at` | TIMESTAMP | NOT NULL (CURRENT_TIMESTAMP) | Upload timestamp |

**Indexes**:
- `ix_org_docs_org_id` on `(organization_id)`

---

### 2.3 Table: `project_categories` (New)
Standardized taxonomy for organizing charitable initiatives:

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | VARCHAR(36) | PRIMARY KEY | UUIDv4 identifier |
| `slug` | VARCHAR(64) | UNIQUE, NOT NULL | URL-safe slug (e.g. `clean-water`, `education`) |
| `name` | VARCHAR(128) | NOT NULL | Human-readable name |
| `description` | TEXT | YES | Category description |
| `icon` | VARCHAR(64) | YES | FontAwesome / UI icon identifier |
| `is_active` | BOOLEAN | NOT NULL, DEFAULT TRUE | Category visibility flag |
| `sort_order` | INTEGER | NOT NULL, DEFAULT 0 | Display ordering weight |

---

### 2.4 Table: `projects` (Altered)
Enriched to support multi-stage lifecycle, currencies, geographic metadata, and reviewer workflows:

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `summary` | TEXT | YES | NULL | Concise 1-2 sentence initiative summary |
| `objectives` | TEXT | YES | NULL | Comprehensive project goals and targets |
| `project_type` | VARCHAR(50) | NO | `'STANDARD'` | Type enum (`STANDARD`, `INFRASTRUCTURE`, `EMERGENCY_RELIEF`, `ENVIRONMENTAL`) |
| `public_visibility`| VARCHAR(20) | NO | `'PUBLIC'` | Visibility enum (`PUBLIC`, `PRIVATE`, `UNLISTED`) |
| `funding_target_minor_units`| INTEGER | NO | 0 | Total funding goal in integer minor units (cents) |
| `currency` | VARCHAR(3) | NO | `'USD'` | ISO 4217 3-letter currency code |
| `country` | VARCHAR(3) | YES | NULL | Country of implementation |
| `location` | VARCHAR(255) | YES | NULL | Geographic locale or city name |
| `submitted_at` | TIMESTAMP | YES | NULL | Timestamp of submission |
| `approved_at` | TIMESTAMP | YES | NULL | Timestamp of reviewer approval |
| `activated_at` | TIMESTAMP | YES | NULL | Timestamp of activation |
| `completed_at` | TIMESTAMP | YES | NULL | Timestamp of project completion |
| `reviewer_id` | VARCHAR(36) | YES | NULL | Foreign key referencing `users.id` |
| `review_notes` | TEXT | YES | NULL | Reviewer notes or feedback |

---

### 2.5 Table: `milestones` (Altered)
Upgraded from a flat placeholder to an evidence-ready, geographically-referenced deliverable specification:

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `objective` | TEXT | YES | NULL | Detailed narrative description |
| `deliverables` | JSON / Text | YES | NULL | Tangible physical/digital outputs |
| `expected_outcome` | TEXT | YES | NULL | Social or environmental metric of success |
| `success_criteria` | TEXT | YES | NULL | Criteria required for reviewer sign-off |
| `target_latitude` | FLOAT | YES | NULL | Latitude coordinate ($-90.0 \le \text{lat} \le 90.0$) |
| `target_longitude` | FLOAT | YES | NULL | Longitude coordinate ($-180.0 \le \text{lng} \le 180.0$) |

**Unique Constraint**:
```sql
CONSTRAINT uq_milestone_project_seq UNIQUE (project_id, sequence_number)
```
Guarantees that no project can contain conflicting or duplicate milestone sequence numbers.

---

## 3. Migration Execution & Rollback Safety

### Upgrade
The upgrade script dynamically checks table structures and applies idempotent alterations:
```bash
alembic upgrade head
```

### Downgrade
The migration implements full, symmetric `downgrade()` logic that safely drops newly introduced tables (`organization_documents`, `project_categories`), removes unique constraints, and removes added columns from `organizations`, `projects`, and `milestones`.
```bash
alembic downgrade -1
```
