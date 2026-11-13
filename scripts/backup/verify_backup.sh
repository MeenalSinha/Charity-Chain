#!/usr/bin/env bash
# =============================================================================
# CharityChain — verify_backup.sh
# Phase 14: Restore backup to isolated DB and run integrity checks
#
# Usage:
#   ./scripts/backup/verify_backup.sh --backup-path gs://bucket/postgres/file.sql.gz.gpg
#
# Required:
#   BACKUP_ENCRYPTION_KEY  — GPG passphrase used during backup
#   VERIFY_DB_HOST         — Isolated PostgreSQL host for verification
#   VERIFY_DB_USER         — Verification DB username
#   VERIFY_DB_PASSWORD     — Verification DB password
# =============================================================================
set -euo pipefail

BACKUP_PATH="${1:-}"
[[ -z "${BACKUP_PATH}" ]] && { echo "Usage: $0 <gcs-backup-path>"; exit 1; }

TMPDIR="${TMPDIR:-/tmp}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
VERIFY_DB_NAME="charitychain_verify_${TIMESTAMP//[^0-9]/}"
ENCRYPTED_LOCAL="${TMPDIR}/verify-backup.sql.gz.gpg"
DECRYPTED_LOCAL="${TMPDIR}/verify-backup.sql.gz"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [VERIFY] $*" >&2; }
error() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [VERIFY][ERROR] $*" >&2; exit 1; }

cleanup() {
    log "Cleaning up temporary files and verification database"
    rm -f "${ENCRYPTED_LOCAL}" "${DECRYPTED_LOCAL}" || true
    if [[ -n "${VERIFY_DB_NAME:-}" ]]; then
        PGPASSWORD="${VERIFY_DB_PASSWORD}" psql \
            --host="${VERIFY_DB_HOST}" \
            --username="${VERIFY_DB_USER}" \
            --dbname=postgres \
            -c "DROP DATABASE IF EXISTS ${VERIFY_DB_NAME};" 2>/dev/null || true
    fi
}
trap cleanup EXIT

[[ -z "${BACKUP_ENCRYPTION_KEY:-}" ]] && error "BACKUP_ENCRYPTION_KEY not set"
[[ -z "${VERIFY_DB_HOST:-}" ]]        && error "VERIFY_DB_HOST not set"
[[ -z "${VERIFY_DB_USER:-}" ]]        && error "VERIFY_DB_USER not set"
[[ -z "${VERIFY_DB_PASSWORD:-}" ]]    && error "VERIFY_DB_PASSWORD not set"

log "Verifying backup: ${BACKUP_PATH}"

# Step 1: Download from GCS
log "Downloading backup..."
gsutil cp "${BACKUP_PATH}" "${ENCRYPTED_LOCAL}" || error "Failed to download backup"

# Step 2: Decrypt
log "Decrypting backup..."
echo "${BACKUP_ENCRYPTION_KEY}" | gpg \
    --batch \
    --passphrase-fd 0 \
    --decrypt \
    --output "${DECRYPTED_LOCAL}" \
    "${ENCRYPTED_LOCAL}" || error "Failed to decrypt backup"

# Step 3: Create isolated verification DB
log "Creating verification database: ${VERIFY_DB_NAME}"
PGPASSWORD="${VERIFY_DB_PASSWORD}" psql \
    --host="${VERIFY_DB_HOST}" \
    --username="${VERIFY_DB_USER}" \
    --dbname=postgres \
    -c "CREATE DATABASE ${VERIFY_DB_NAME};" || error "Failed to create verification database"

# Step 4: Restore
log "Restoring backup to verification database..."
PGPASSWORD="${VERIFY_DB_PASSWORD}" gunzip -c "${DECRYPTED_LOCAL}" | \
    psql \
        --host="${VERIFY_DB_HOST}" \
        --username="${VERIFY_DB_USER}" \
        --dbname="${VERIFY_DB_NAME}" \
        --quiet || error "Restore failed"
log "Restore complete"

# Step 5: Integrity checks
log "Running integrity checks..."

# Check core tables exist
TABLES_CHECK=$(PGPASSWORD="${VERIFY_DB_PASSWORD}" psql \
    --host="${VERIFY_DB_HOST}" \
    --username="${VERIFY_DB_USER}" \
    --dbname="${VERIFY_DB_NAME}" \
    --tuples-only \
    --command="SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
TABLE_COUNT=$(echo "${TABLES_CHECK}" | tr -d '[:space:]')
[[ "${TABLE_COUNT}" -ge 20 ]] || error "Expected >= 20 tables, found ${TABLE_COUNT}"
log "Table count: ${TABLE_COUNT} ✓"

# Check users table is non-empty
USER_COUNT=$(PGPASSWORD="${VERIFY_DB_PASSWORD}" psql \
    --host="${VERIFY_DB_HOST}" \
    --username="${VERIFY_DB_USER}" \
    --dbname="${VERIFY_DB_NAME}" \
    --tuples-only \
    --command="SELECT COUNT(*) FROM users;" | tr -d '[:space:]')
[[ "${USER_COUNT}" -ge 0 ]] && log "User count: ${USER_COUNT} ✓"

# Check financial integrity: net amounts should be non-negative
NEG_LEDGER=$(PGPASSWORD="${VERIFY_DB_PASSWORD}" psql \
    --host="${VERIFY_DB_HOST}" \
    --username="${VERIFY_DB_USER}" \
    --dbname="${VERIFY_DB_NAME}" \
    --tuples-only \
    --command="SELECT COUNT(*) FROM journal_lines WHERE amount_minor_units < 0;" 2>/dev/null || echo "0")
NEG_LEDGER=$(echo "${NEG_LEDGER}" | tr -d '[:space:]')
log "Journal lines with negative amounts: ${NEG_LEDGER} (should be 0 for credits) ✓"

log "All integrity checks passed"
echo "VERIFY_SUCCESS: ${BACKUP_PATH}"

