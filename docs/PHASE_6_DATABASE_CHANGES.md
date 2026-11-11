# Phase 6 Database Changes & Schema Evolution — CharityChain

## 1. Alembic Migration Overview

- **Migration ID**: `006_phase6_ai_verification`
- **Revision File**: `backend/alembic/versions/006_phase6_ai_verification.py`
- **Reversibility**: Fully reversible (`upgrade head` -> `downgrade -1` -> `upgrade head` verified).

---

## 2. Table Schemas Modified & Added

### 2.1 Extended Table: `verification_reports`

The existing `verification_reports` table from earlier phases was updated with structured, explainable columns:

| Column Name | Type | Nullable | Description |
|-------------|------|----------|-------------|
| `pipeline_version` | String(32) | No | Version of the verification pipeline (e.g. `1.0.0`) |
| `overall_assessment` | String(32) | No | Strict enum: `CONSISTENT`, `INCONCLUSIVE`, `FLAGGED`, `FAILED` |
| `confidence_score` | Float | Yes | Aggregate multi-indicator confidence score ($[0.0, 1.0]$) |
| `calibration_status` | String(32) | No | Default: `"UNVALIDATED"` |
| `processing_started_at` | DateTime | Yes | Start timestamp of pipeline execution |
| `processing_completed_at` | DateTime | Yes | Completion timestamp of pipeline execution |
| `model_metadata_json` | JSON | Yes | Model variant, weights hash, and runtime execution flags |
| `summary_notes` | Text | Yes | Grounded explainable summary of findings |
| `limitations_json` | JSON | Yes | Known boundaries, assumptions, and disclosures |
| `config_snapshot_json` | JSON | Yes | Active threshold settings snapshot at time of execution |

---

### 2.2 Extended Table: `verification_check_records`

Individual atomic check records linked to `verification_reports`:

| Column Name | Type | Nullable | Description |
|-------------|------|----------|-------------|
| `check_type` | String(64) | No | Indicator name (`FILE_INTEGRITY`, `OBJECT_DETECTION`, `ERROR_LEVEL_ANALYSIS`, etc.) |
| `status` | String(32) | No | Status enum: `PASSED`, `FLAGGED`, `FAILED`, `NOT_APPLICABLE` |
| `passed` | Boolean | No | Boolean flag indicating whether check was satisfied |
| `message` | Text | Yes | Human-readable explanation of check outcome |
| `technical_result_json` | JSON | Yes | Structured result payload from analyzer |
| `raw_metrics_json` | JSON | Yes | Raw numeric metrics (e.g. mean error, SSIM score, pHash distance) |
| `model_name` | String(64) | Yes | Name of executing model/analyzer |
| `model_version` | String(32) | Yes | Version of executing model |
| `duration_ms` | Float | Yes | Execution duration in milliseconds |
| `diagnostic_artifact_path`| String(512) | Yes | Relative storage path to generated visual artifact |

---

### 2.3 New Table: `verification_jobs`

Manages asynchronous queue processing and durable state tracking:

```sql
CREATE TABLE verification_jobs (
    id VARCHAR(36) PRIMARY KEY,
    evidence_file_id VARCHAR(36) NOT NULL REFERENCES evidence_files(id) ON DELETE CASCADE,
    baseline_evidence_id VARCHAR(36) REFERENCES evidence_files(id) ON DELETE SET NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'QUEUED',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    priority INTEGER NOT NULL DEFAULT 0,
    requested_by_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    report_id VARCHAR(36) REFERENCES verification_reports(id) ON DELETE SET NULL,
    error_category VARCHAR(64),
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_verification_jobs_status ON verification_jobs(status);
CREATE INDEX ix_verification_jobs_evidence_id ON verification_jobs(evidence_file_id);
```

---

## 3. Migration Integrity Verification

The migration was tested bidirectionally on SQLite and PostgreSQL:
```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```
All schema transitions execute idempotently without data loss or residual dangling indexes.
