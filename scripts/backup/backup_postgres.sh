#!/usr/bin/env bash
# =============================================================================
# CharityChain — backup_postgres.sh
# Phase 14: Automated PostgreSQL backup with encryption and GCS upload
#
# Usage:
#   ./scripts/backup/backup_postgres.sh [--env production|staging]
#
# Required environment variables:
#   BACKUP_BUCKET    — GCS bucket for backups (e.g. charitychain-backups-prod-PROJECT)
#   POSTGRES_HOST    — PostgreSQL host
#   POSTGRES_PORT    — PostgreSQL port (default: 5432)
#   POSTGRES_USER    — Database username
#   POSTGRES_PASSWORD — Database password
#   POSTGRES_DB      — Database name
#   BACKUP_ENCRYPTION_KEY — GPG symmetric passphrase for backup encryption
#
# Optional:
#   RETENTION_DAYS   — Days to keep backups (default: 90)
#   BACKUP_ENV       — Environment label (default: production)
# =============================================================================
set -euo pipefail

# --- Configuration ---
ENV="${BACKUP_ENV:-production}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_FILENAME="charitychain-${ENV}-${TIMESTAMP}.sql.gz"
ENCRYPTED_FILENAME="${BACKUP_FILENAME}.gpg"
TMPDIR="${TMPDIR:-/tmp}"
BACKUP_PATH="${TMPDIR}/${BACKUP_FILENAME}"
ENCRYPTED_PATH="${TMPDIR}/${ENCRYPTED_FILENAME}"
GCS_PATH="gs://${BACKUP_BUCKET}/postgres/${ENCRYPTED_FILENAME}"
RETENTION_DAYS="${RETENTION_DAYS:-90}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [BACKUP] $*" >&2
}

error() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [BACKUP][ERROR] $*" >&2
    exit 1
}

cleanup() {
    log "Cleaning up temporary files"
    rm -f "${BACKUP_PATH}" "${ENCRYPTED_PATH}" || true
}
trap cleanup EXIT

# --- Prerequisite checks ---
command -v pg_dump >/dev/null 2>&1 || error "pg_dump not found — install postgresql-client"
command -v gpg     >/dev/null 2>&1 || error "gpg not found — install gnupg"
command -v gsutil  >/dev/null 2>&1 || error "gsutil not found — install google-cloud-sdk"

[[ -z "${BACKUP_BUCKET:-}"           ]] && error "BACKUP_BUCKET is not set"
[[ -z "${POSTGRES_HOST:-}"           ]] && error "POSTGRES_HOST is not set"
[[ -z "${POSTGRES_USER:-}"           ]] && error "POSTGRES_USER is not set"
[[ -z "${POSTGRES_PASSWORD:-}"       ]] && error "POSTGRES_PASSWORD is not set"
[[ -z "${POSTGRES_DB:-}"             ]] && error "POSTGRES_DB is not set"
[[ -z "${BACKUP_ENCRYPTION_KEY:-}"   ]] && error "BACKUP_ENCRYPTION_KEY is not set"

log "Starting backup: ${BACKUP_FILENAME}"
log "Environment: ${ENV}"
log "Target bucket: ${BACKUP_BUCKET}"

# --- Step 1: pg_dump ---
log "Running pg_dump..."
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    --host="${POSTGRES_HOST}" \
    --port="${POSTGRES_PORT}" \
    --username="${POSTGRES_USER}" \
    --dbname="${POSTGRES_DB}" \
    --format=plain \
    --no-password \
    --verbose \
    | gzip --best > "${BACKUP_PATH}"

BACKUP_SIZE="$(du -sh "${BACKUP_PATH}" | cut -f1)"
log "pg_dump complete — compressed size: ${BACKUP_SIZE}"

# --- Step 2: Encrypt with GPG symmetric encryption ---
log "Encrypting backup..."
echo "${BACKUP_ENCRYPTION_KEY}" | gpg \
    --batch \
    --yes \
    --passphrase-fd 0 \
    --symmetric \
    --cipher-algo AES256 \
    --output "${ENCRYPTED_PATH}" \
    "${BACKUP_PATH}"

ENCRYPTED_SIZE="$(du -sh "${ENCRYPTED_PATH}" | cut -f1)"
log "Encryption complete — encrypted size: ${ENCRYPTED_SIZE}"

# --- Step 3: Upload to GCS ---
log "Uploading to ${GCS_PATH}..."
gsutil -q cp "${ENCRYPTED_PATH}" "${GCS_PATH}"
log "Upload complete"

# --- Step 4: Verify upload ---
log "Verifying upload..."
gsutil stat "${GCS_PATH}" >/dev/null || error "Upload verification failed — file not found in GCS"
log "Upload verified"

# --- Step 5: Apply retention policy (delete old backups) ---
log "Applying retention policy (${RETENTION_DAYS} days)..."
CUTOFF_DATE="$(date -u -d "${RETENTION_DAYS} days ago" +%Y%m%d 2>/dev/null || date -u -v-"${RETENTION_DAYS}"d +%Y%m%d)"
gsutil ls "gs://${BACKUP_BUCKET}/postgres/" 2>/dev/null | while read -r obj; do
    obj_date="$(basename "${obj}" | grep -oP '\d{8}' | head -1 || true)"
    if [[ -n "${obj_date}" && "${obj_date}" < "${CUTOFF_DATE}" ]]; then
        log "Deleting expired backup: ${obj}"
        gsutil rm "${obj}" || log "Warning: could not delete ${obj}"
    fi
done

# --- Step 6: Record metadata ---
METADATA_FILE="${TMPDIR}/backup-metadata-${TIMESTAMP}.json"
cat > "${METADATA_FILE}" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "environment": "${ENV}",
  "database": "${POSTGRES_DB}",
  "host": "${POSTGRES_HOST}",
  "filename": "${ENCRYPTED_FILENAME}",
  "gcs_path": "${GCS_PATH}",
  "compressed_size": "${BACKUP_SIZE}",
  "encrypted_size": "${ENCRYPTED_SIZE}",
  "retention_days": ${RETENTION_DAYS},
  "status": "SUCCESS"
}
EOF
gsutil cp "${METADATA_FILE}" "gs://${BACKUP_BUCKET}/metadata/backup-${TIMESTAMP}.json"
rm -f "${METADATA_FILE}"

log "Backup completed successfully: ${GCS_PATH}"
echo "BACKUP_SUCCESS: ${GCS_PATH}"

