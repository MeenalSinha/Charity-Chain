#!/usr/bin/env bash
# =============================================================================
# CharityChain — run_migrations.sh
# Phase 14: Safe database migration runner with pre-flight backup
#
# Usage:
#   ./scripts/deploy/run_migrations.sh [--env production|staging|development]
#
# Required environment variables:
#   DATABASE_URL — SQLAlchemy database URL
# =============================================================================
set -euo pipefail

DEPLOY_ENV="${DEPLOY_ENV:-development}"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [MIGRATE] $*"; }
error() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [MIGRATE][ERROR] $*" >&2; exit 1; }

[[ -z "${DATABASE_URL:-}" ]] && error "DATABASE_URL must be set"

log "Environment: ${DEPLOY_ENV}"
log "Checking current migration state..."
alembic current

log "Checking for pending migrations..."
PENDING=$(alembic check 2>&1 || true)
if echo "${PENDING}" | grep -q "Target database is not up to date"; then
    log "Pending migrations detected"
else
    log "Database is already at latest revision"
fi

# Pre-migration backup in production only
if [[ "${DEPLOY_ENV}" == "production" ]]; then
    log "PRODUCTION: Running pre-migration backup..."
    if [[ -f "./scripts/backup/backup_postgres.sh" ]]; then
        ./scripts/backup/backup_postgres.sh || error "Pre-migration backup failed — aborting migration"
        log "Pre-migration backup complete"
    else
        log "WARNING: backup script not found — confirm manual backup was taken before proceeding"
        read -r -p "Confirm manual backup has been taken? (yes/no): " CONFIRM
        [[ "${CONFIRM}" == "yes" ]] || error "Migration aborted — backup not confirmed"
    fi
fi

log "Applying migrations to HEAD..."
alembic upgrade head

log "Verifying final migration state..."
alembic current

log "Migration completed successfully."

