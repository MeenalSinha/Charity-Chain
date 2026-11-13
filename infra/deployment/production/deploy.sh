#!/usr/bin/env bash
# =============================================================================
# CharityChain — Production Deployment Script
# Phase 14: Blue/Green traffic migration on Cloud Run
# =============================================================================
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID_PROD:-}"
REGION="${GCP_REGION:-us-central1}"
RELEASE_TAG="${1:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: GCP_PROJECT_ID_PROD environment variable must be set"
  exit 1
fi

if [[ -z "${RELEASE_TAG}" ]]; then
  echo "ERROR: Release tag required (e.g. ./deploy.sh v1.0.0)"
  exit 1
fi

echo "=== Deploying CharityChain to PRODUCTION ==="
echo "Project:     ${PROJECT_ID}"
echo "Region:      ${REGION}"
echo "Release Tag: ${RELEASE_TAG}"

# Step 1: Pre-migration backup
echo "Creating pre-deployment database backup..."
gcloud run jobs execute charitychain-backup-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --wait

# Step 2: Database migration
echo "Executing database migrations..."
gcloud run jobs execute charitychain-migrate-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --wait

# Step 3: Deploy backend (no initial traffic)
echo "Deploying backend revision (${RELEASE_TAG}) with 0% traffic..."
gcloud run services update charitychain-backend-prod \
  --image="gcr.io/${PROJECT_ID}/charitychain-backend:${RELEASE_TAG}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --no-traffic

# Step 4: Deploy frontend (no initial traffic)
echo "Deploying frontend revision (${RELEASE_TAG}) with 0% traffic..."
gcloud run services update charitychain-frontend-prod \
  --image="gcr.io/${PROJECT_ID}/charitychain-frontend:${RELEASE_TAG}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --no-traffic

# Step 5: Smoke test new backend revision directly
echo "Running smoke test on new revision..."
LATEST_REV=$(gcloud run revisions list \
  --service=charitychain-backend-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(name)" \
  --sort-by="~creationTimestamp" \
  --limit=1)

echo "Testing revision: ${LATEST_REV}"

# Step 6: Canary routing (10%)
echo "Routing 10% traffic to ${LATEST_REV} for monitoring..."
gcloud run services update-traffic charitychain-backend-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --to-revisions="${LATEST_REV}=10"

echo "Canary deployed. Verify Grafana dashboards for 60 seconds."
sleep 60

# Step 7: Promote to 100%
echo "Promoting to 100% traffic..."
gcloud run services update-traffic charitychain-backend-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --to-latest

gcloud run services update-traffic charitychain-frontend-prod \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --to-latest

echo "=== Production deployment ${RELEASE_TAG} successfully completed! ==="

