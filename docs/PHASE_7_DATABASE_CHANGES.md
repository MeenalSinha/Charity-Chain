# CharityChain — Phase 7 Database Changes & Migration Report

## 1. Overview

Phase 7 introduces 7 database tables, schema extensions, and unique constraints to support human review cases, dual reviewer task assignments, immutable decision records, formal disputes, discussion threads, and cryptographically chained audit events.

All database schema evolutions are managed via Alembic migration `007_phase7_human_review_workflows.py`.

---

## 2. Table Specifications

### 2.1 `review_cases`
Represents an NGO's formal submission of a milestone evidence package for human evaluation.
- `id`: VARCHAR(36), Primary Key
- `milestone_id`: VARCHAR(36), FK -> `milestones.id` (CASCADE), Indexed
- `project_id`: VARCHAR(36), FK -> `projects.id` (CASCADE), Indexed
- `organization_id`: VARCHAR(36), FK -> `organizations.id` (CASCADE), Indexed
- `submission_round`: INTEGER, Default 1, Indexed
- `submitted_by_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Indexed
- `status`: VARCHAR(50), Default 'SUBMITTED', Indexed
- `final_decision`: VARCHAR(50), Nullable (APPROVE, REJECT, REQUEST_REVISION)
- `reconciliation_result`: VARCHAR(50), Nullable
- `submission_notes`: TEXT, Nullable
- `snapshot_digest`: VARCHAR(64), Nullable (SHA-256 of frozen evidence package)
- `snapshot_data_json`: JSON, Nullable
- `review_policy_version`: VARCHAR(32), Default '1.0'
- `revision_deadline`: DATETIME, Nullable
- `appeal_deadline`: DATETIME, Nullable
- `finalized_at`: DATETIME, Nullable
- Standard timestamps (`created_at`, `updated_at`, `is_active`)

### 2.2 `review_case_evidence`
Binds review cases to exact evidence files, version numbers, SHA-256 hashes, and Phase 6 AI verification reports.
- `id`: VARCHAR(36), Primary Key
- `review_case_id`: VARCHAR(36), FK -> `review_cases.id` (CASCADE), Indexed
- `evidence_file_id`: VARCHAR(36), FK -> `evidence_files.id` (CASCADE), Indexed
- `evidence_version`: INTEGER, Not Null
- `evidence_hash`: VARCHAR(64), Not Null
- `verification_report_id`: VARCHAR(36), FK -> `verification_reports.id` (SET NULL), Nullable
- `ai_pipeline_version`: VARCHAR(32), Nullable
- `evidence_role`: VARCHAR(64), Default 'PRIMARY'
- `snapshot_timestamp`: DATETIME, Default utcnow

### 2.3 `review_tasks`
Tracks individual reviewer assignments, task slots (Slot 1 and Slot 2), and acceptance lifecycle.
- `id`: VARCHAR(36), Primary Key
- `review_case_id`: VARCHAR(36), FK -> `review_cases.id` (CASCADE), Indexed
- `milestone_id`: VARCHAR(36), FK -> `milestones.id` (CASCADE), Indexed
- `assigned_reviewer_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Indexed
- `reviewer_slot`: INTEGER, Default 1 (1 = Primary, 2 = Secondary)
- `status`: VARCHAR(50), Default 'PENDING'
- `assignment_state`: VARCHAR(50), Default 'ASSIGNED'
- `assigned_at`: DATETIME, Default utcnow
- `accepted_at`: DATETIME, Nullable
- `due_at`: DATETIME, Nullable
- `decided_at`: DATETIME, Nullable
- `decline_reason`: TEXT, Nullable
- `conflict_declared`: BOOLEAN, Default False
- `conflict_reason`: TEXT, Nullable
- `reassignment_count`: INTEGER, Default 0

### 2.4 `review_decisions`
Stores immutable review decisions and comments submitted for each task slot.
- `id`: VARCHAR(36), Primary Key
- `review_case_id`: VARCHAR(36), FK -> `review_cases.id` (CASCADE), Indexed
- `task_id`: VARCHAR(36), FK -> `review_tasks.id` (CASCADE), Indexed
- `reviewer_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Indexed
- `decision`: SQLEnum(ReviewDecision), Not Null
- `comments`: TEXT, Not Null
- `structured_revision_data_json`: JSON, Nullable
- `rejection_reason_category`: VARCHAR(100), Nullable
- `evidence_snapshot_digest`: VARCHAR(64), Nullable
- `submission_round`: INTEGER, Default 1
- `is_superseded`: BOOLEAN, Default False
- `decided_at`: DATETIME, Default utcnow

### 2.5 `review_disputes`
Manages formal appeals filed by NGOs against rejected review cases.
- `id`: VARCHAR(36), Primary Key
- `review_case_id`: VARCHAR(36), FK -> `review_cases.id` (CASCADE), Indexed
- `milestone_id`: VARCHAR(36), FK -> `milestones.id` (CASCADE), Indexed
- `project_id`: VARCHAR(36), FK -> `projects.id` (CASCADE), Indexed
- `organization_id`: VARCHAR(36), FK -> `organizations.id` (CASCADE), Indexed
- `original_decision_record_id`: VARCHAR(36), FK -> `review_decisions.id` (SET NULL), Nullable
- `raised_by_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Indexed
- `assigned_adjudicator_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Nullable
- `status`: VARCHAR(50), Default 'OPEN'
- `dispute_category`: VARCHAR(100), Not Null
- `dispute_reason`: TEXT, Not Null
- `supporting_evidence_json`: JSON, Nullable
- `resolution`: VARCHAR(100), Nullable
- `resolution_explanation`: TEXT, Nullable
- `resolved_at`: DATETIME, Nullable

### 2.6 `review_dispute_messages`
Stores threaded discussion messages between the NGO and the assigned adjudicator.
- `id`: VARCHAR(36), Primary Key
- `dispute_id`: VARCHAR(36), FK -> `review_disputes.id` (CASCADE), Indexed
- `sender_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Indexed
- `sender_role`: VARCHAR(50), Not Null
- `message`: TEXT, Not Null
- `attachments_json`: JSON, Nullable
- Standard timestamps (`created_at`, `updated_at`, `is_active`)

### 2.7 `review_audit_events`
Immutable, append-only cryptographic ledger tracking every review lifecycle event.
- `id`: VARCHAR(36), Primary Key
- `review_case_id`: VARCHAR(36), FK -> `review_cases.id` (SET NULL), Nullable, Indexed
- `milestone_id`: VARCHAR(36), FK -> `milestones.id` (CASCADE), Indexed
- `actor_id`: VARCHAR(36), FK -> `users.id` (SET NULL), Nullable, Indexed
- `event_type`: VARCHAR(50), Not Null, Indexed
- `details_json`: JSON, Nullable
- `previous_event_hash`: VARCHAR(64), Not Null
- `current_event_hash`: VARCHAR(64), Not Null, Unique, Indexed
- `created_at`: DATETIME, Default utcnow

---

## 3. Migration Reversibility Verification

The Alembic migration script `007_phase7_human_review_workflows.py` was bidirectionally tested on SQLite:
1. `alembic upgrade head`: Applied migration cleanly with exit code 0.
2. `alembic downgrade -1`: Rolled back all 7 tables and indexes cleanly with exit code 0.
3. `alembic upgrade head`: Re-applied migration cleanly with exit code 0.
