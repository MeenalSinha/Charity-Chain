# CharityChain — Phase 11 Database Changes

## 1. Overview
Phase 11 introduces three new analytics tables, indexes for performance, and extends the `projects` and `milestones` tables to support location sensitivity and beneficiary safety.

All changes are managed via Alembic migration `011_phase11_analytics_and_transparency.py`.

---

## 2. Table Schemas

### 2.1 `analytics_snapshots`
Caches point-in-time analytical aggregates to accelerate dashboard load times and facilitate historical trend analysis.
```sql
CREATE TABLE analytics_snapshots (
    id VARCHAR(36) PRIMARY KEY,
    snapshot_type VARCHAR(64) NOT NULL,
    scope_id VARCHAR(36) NULL,
    snapshot_date DATE NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_analytics_snapshots_type ON analytics_snapshots(snapshot_type);
CREATE INDEX ix_analytics_snapshots_scope ON analytics_snapshots(scope_id);
CREATE INDEX ix_analytics_snapshots_date ON analytics_snapshots(snapshot_date);
```

### 2.2 `analytics_export_jobs`
Tracks user-requested asynchronous analytics data exports.
```sql
CREATE TABLE analytics_export_jobs (
    id VARCHAR(36) PRIMARY KEY,
    requested_by_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    export_type VARCHAR(64) NOT NULL,
    file_format VARCHAR(16) NOT NULL DEFAULT 'CSV',
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    file_storage_key VARCHAR(255) NULL,
    checksum_sha256 VARCHAR(64) NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITHOUT TIME ZONE NULL
);
CREATE INDEX ix_analytics_export_jobs_user ON analytics_export_jobs(requested_by_id);
CREATE INDEX ix_analytics_export_jobs_status ON analytics_export_jobs(status);
```

### 2.3 `public_transparency_publications`
Immutable historical record of periodic public transparency reports published by platform administrators.
```sql
CREATE TABLE public_transparency_publications (
    id VARCHAR(36) PRIMARY KEY,
    published_by_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    title VARCHAR(255) NOT NULL,
    publication_type VARCHAR(64) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    report_hash VARCHAR(64) NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_transparency_pub_type ON public_transparency_publications(publication_type);
CREATE INDEX ix_transparency_pub_hash ON public_transparency_publications(report_hash);
```

---

## 3. Schema Extensions

### 3.1 `projects` Table
Added location privacy column:
```sql
ALTER TABLE projects ADD COLUMN is_location_sensitive BOOLEAN NOT NULL DEFAULT FALSE;
```

### 3.2 `milestones` Table
Added location privacy column:
```sql
ALTER TABLE milestones ADD COLUMN is_location_sensitive BOOLEAN NOT NULL DEFAULT FALSE;
```

---

## 4. Reversibility & Data Integrity
- The Alembic migration `011_phase11_analytics_and_transparency.py` implements both `upgrade()` and `downgrade()` functions.
- Downgrade cleanly drops the three tables and removes the `is_location_sensitive` columns from `projects` and `milestones`.

