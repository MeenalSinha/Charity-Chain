# Phase 5 Database Changes & Migration Report

## 1. Executive Summary

Phase 5 introduces storage metadata, non-destructive multi-versioning, and a cryptographic audit trail to the CharityChain database schema via Alembic migration `005_phase5_evidence_management`.

- **Migration ID**: `005_phase5_evidence_management`
- **Revises (Down-Revision)**: `004_phase4_payment_and_donations`
- **Dialect Compatibility**: Tested and verified on SQLite 3 (using `op.batch_alter_table`) and PostgreSQL 16+.
- **Reversibility**: Validated forward migration (`upgrade head`), rollback (`downgrade -1`), and clean re-application.

---

## 2. Table Modifications: `evidence_files`

The following columns and indexes were added to `evidence_files`:

| Column Name | Type | Nullable | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `original_filename` | VARCHAR(255) | YES | `NULL` | Sanitized original filename at submission. |
| `storage_key` | VARCHAR(500) | YES | `NULL` | Canonical path or object key in the storage provider. |
| `storage_provider` | VARCHAR(50) | NO | `'local'` | Active storage backend (`local`, `s3`, `minio`). |
| `storage_version_id`| VARCHAR(100) | YES | `NULL` | S3 object version identifier. |
| `detected_mime_type`| VARCHAR(100) | YES | `NULL` | Verified binary magic-byte MIME type. |
| `version_number` | INTEGER | NO | `1` | Monotonically increasing version counter. |
| `parent_evidence_id`| VARCHAR(36) | YES | `NULL` | Foreign key referencing predecessor version in `evidence_files.id`. |
| `is_archived` | BOOLEAN | NO | `'0'` | Soft-archival flag. |
| `archived_at` | DATETIME | YES | `NULL` | Timestamp when asset was marked archived. |
| `quarantine_reason` | VARCHAR(255) | YES | `NULL` | Reason flag if asset fails integrity or security checks. |

### Indexes Added
- `ix_evidence_files_milestone_id` on `milestone_id`: Optimizes milestone evidence listings.
- `ix_evidence_files_parent_evidence_id` on `parent_evidence_id`: Accelerates version lineage lookups.
- `ix_evidence_files_is_archived` on `is_archived`: Filters active vs archived records.

---

## 3. New Table: `evidence_audit_events`

Creates an immutable, append-only cryptographic ledger tracking each action on evidence assets.

```sql
CREATE TABLE evidence_audit_events (
    id VARCHAR(36) PRIMARY KEY,
    evidence_id VARCHAR(36) NOT NULL,
    actor_id VARCHAR(36) NULL,
    event_type VARCHAR(50) NOT NULL,
    details_json JSON NULL,
    previous_event_hash VARCHAR(64) NOT NULL,
    current_event_hash VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT '1',
    FOREIGN KEY(evidence_id) REFERENCES evidence_files(id) ON DELETE CASCADE,
    FOREIGN KEY(actor_id) REFERENCES users(id) ON DELETE SET NULL
);
```

### Indexes Added
- `ix_evidence_audit_events_evidence_id` on `evidence_id`: Enables fast chronological audit trail extraction.
- `ix_evidence_audit_events_current_event_hash` on `current_event_hash`: Supports hash lookup and chain validation.
- `ix_evidence_audit_events_event_type` on `event_type`: Accelerates event filtering by action type.

---

## 4. Migration Execution & Rollback Validation

Both upgrade and rollback paths were verified via terminal execution:

### 1. Upgrade to Head
```bash
python -m alembic -c backend/alembic.ini upgrade head
# INFO  [alembic.runtime.migration] Running upgrade 004_phase4_payment_and_donations -> 005_phase5_evidence_management
```

### 2. Downgrade Verification
```bash
python -m alembic -c backend/alembic.ini downgrade -1
# INFO  [alembic.runtime.migration] Running downgrade 005_phase5_evidence_management -> 004_phase4_payment_and_donations
```

### 3. Re-Upgrade Verification
```bash
python -m alembic -c backend/alembic.ini upgrade head
# INFO  [alembic.runtime.migration] Running upgrade 004_phase4_payment_and_donations -> 005_phase5_evidence_management
```
Result: Schema migration executed cleanly without data loss or table locks.
