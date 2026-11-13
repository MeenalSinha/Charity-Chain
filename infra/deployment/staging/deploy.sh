#!/usr/bin/env bash
# =============================================================================
# CharityChain — Staging Deployment Script
# Phase 14: Automated staging deployment to Cloud Run
# =============================================================================
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID_STAGING:-}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_TAG="${1:-latest}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: GCP_PROJECT_ID_STAGING environment variable must be set"
  exit 1
fi

echo "=== Deploying CharityChain to STAGING ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Tag:     ${IMAGE_TAG}"

# Step 1: Run DB migrations
echo "Running migrations..."
gcloud run jobs execute charitychain-migrate-staging \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --wait

# Step 2: Deploy Backend
echo "Deploying backend..."
gcloud run services update charitychain-backend-staging \
  --image="gcr.io/${PROJECT_ID}/charitychain-backend:${IMAGE_TAG}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

# Step 3: Deploy Frontend
echo "Deploying frontend..."
gcloud run services update charitychain-frontend-staging \
  --image="gcr.io/${PROJECT_ID}/charitychain-frontend:${IMAGE_TAG}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}"

echo "=== Staging deployment complete ==="

