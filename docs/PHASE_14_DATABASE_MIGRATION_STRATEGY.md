# Phase 14 — Database Migration Strategy

**Version:** 1.0  
**Date:** 2026-11-14

---

## 1. Principles

1. **Never run migrations directly in production without a backup**: Every production migration must be preceded by a confirmed backup.
2. **Alembic is the single source of truth**: All schema changes must go through Alembic. Manual DDL changes are forbidden.
3. **Migrations must be reversible**: Every migration must have a working `downgrade()` function.
4. **Zero-downtime is preferred**: Use additive migrations (ADD COLUMN nullable, CREATE TABLE) wherever possible.
5. **Test migrations in staging first**: Migrations must succeed in staging before production.

---

## 2. Migration File Standards

Every migration file must:

```python
"""Phase N: description of changes

Revision ID: XXX
Revises: YYY
Create Date: YYYY-MM-DD

This migration:
- Creates table X
- Adds column Y to table Z (nullable, default: NULL)
- Adds index on (table.column)

Reversible: Yes
Estimated downtime: None (additive migration)
"""
```

---

## 3. Deployment-Safe Migration Patterns

### 3.1 Adding a Nullable Column (Zero Downtime)

```python
def upgrade():
    op.add_column('table_name',
        sa.Column('new_column', sa.String(255), nullable=True)
    )

def downgrade():
    op.drop_column('table_name', 'new_column')
```

### 3.2 Adding a Non-Nullable Column (Requires Data Backfill)

```python
def upgrade():
    # Step 1: Add as nullable
    op.add_column('table_name',
        sa.Column('new_column', sa.String(255), nullable=True)
    )
    # Step 2: Backfill existing rows
    op.execute("UPDATE table_name SET new_column = 'default_value' WHERE new_column IS NULL")
    # Step 3: Set NOT NULL constraint
    op.alter_column('table_name', 'new_column', nullable=False)

def downgrade():
    op.drop_column('table_name', 'new_column')
```

### 3.3 Renaming a Column (Two-Phase Migration)

**Phase A (deploy before code change):**
```python
def upgrade():
    op.add_column('table', sa.Column('new_name', sa.String(255), nullable=True))
    op.execute("UPDATE table SET new_name = old_name")
```

**Phase B (deploy after code change, old_name no longer used):**
```python
def upgrade():
    op.drop_column('table', 'old_name')
```

---

## 4. Migration Execution Script

```bash
#!/usr/bin/env bash
# scripts/deploy/run_migrations.sh
set -euo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [MIGRATE] $*"; }

# --- Preflight ---
log "Checking current migration state..."
alembic current

log "Checking pending migrations..."
alembic check || { log "Pending migrations detected"; }

# --- Pre-migration backup (production only) ---
if [[ "${ENVIRONMENT:-development}" == "production" ]]; then
    log "Running pre-migration backup..."
    ./scripts/backup/backup_postgres.sh
    log "Backup complete. Proceeding with migration."
fi

# --- Apply migrations ---
log "Applying migrations..."
alembic upgrade head

log "Verifying migration state..."
alembic current

log "Migrations completed successfully."
```

---

## 5. Current Migration Chain

| Revision | Phase | Description |
|:---|:---|:---|
| `001_initial_schema` | Phase 1 | Base user and organization tables |
| `002_phase2_auth_sessions` | Phase 2 | Auth sessions, audit events |
| `003_phase3_org_and_projects` | Phase 3 | Org documents, projects, milestones |
| `004_phase4_payment_and_donations` | Phase 4 | Donations, payments, webhooks |
| `005_phase5_evidence_management` | Phase 5 | Evidence files, audit events |
| `006_phase6_ai_verification` | Phase 6 | Verification reports, jobs |
| `007_phase7_human_review_workflows` | Phase 7 | Review cases, tasks, disputes |
| `008_phase8_disbursement_and_financial_ledger` | Phase 8 | Payouts, ledger, reconciliation |
| `009_phase9_proof_of_impact` | Phase 9 | Certificates, impact ledger |
| `010_phase10_notifications_and_webhooks` | Phase 10 | Outbox, notifications, webhooks |
| `011_phase11_analytics` | Phase 11 | Analytics snapshots, transparency |
| `012_phase13_security_audit_and_observability` | Phase 13 | Security audit events |

---

## 6. Emergency Rollback Procedure

If a migration causes issues in production:

```bash
# Roll back one revision
alembic downgrade -1

# Roll back to specific revision
alembic downgrade 011_phase11_analytics

# NEVER: Roll back to base (would destroy all data)
# alembic downgrade base  ← FORBIDDEN
```

**After emergency rollback:**
1. Document the specific error
2. Fix the migration script
3. Test the fix in staging
4. Re-apply in production during next maintenance window

