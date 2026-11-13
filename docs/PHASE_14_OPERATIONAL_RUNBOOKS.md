# Phase 14 — Operational Runbooks

**Version:** 1.0  
**Date:** 2026-11-14  
**Audience:** On-call engineers, DevOps, Platform team

---

## RB-01: Responding to 5xx Errors (Backend)

**Trigger**: Error rate > 1% over 5 minutes, or alert from Sentry/Grafana.

**Steps:**
1. Check Sentry for error details: https://sentry.io/organizations/charitychain/issues/
2. Check Cloud Logging:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=50 --format=json
   ```
3. Check metrics on Grafana `app-health` dashboard
4. If new deployment caused regression: Rollback (see RB-02)
5. If database connection issue: Check Cloud SQL connectivity (see RB-07)
6. If memory/CPU spike: Check Cloud Run metrics; scale replicas if needed
7. Document incident in incident log

---

## RB-02: Rolling Back a Failed Deployment

**Trigger**: New deployment causes errors or health check failures.

**Steps:**
```bash
# 1. List recent revisions
gcloud run revisions list --service=charitychain-backend-prod \
  --region=us-central1 --format="table(name,creationTimestamp,status.conditions)"

# 2. Identify the last known-good revision
PREV_REV=charitychain-backend-prod-XXXXX

# 3. Route 100% traffic to previous revision
gcloud run services update-traffic charitychain-backend-prod \
  --region=us-central1 \
  --to-revisions="${PREV_REV}=100"

# 4. Verify rollback
curl -f https://api.charitychain.dev/health
```

---

## RB-03: Rotating Secrets

**Trigger**: Scheduled rotation (quarterly) or suspected compromise.

**Steps:**
```bash
# 1. Generate new secret
NEW_SECRET=$(python -c "import secrets; print(secrets.token_hex(64))")

# 2. Add new version to Secret Manager
echo -n "${NEW_SECRET}" | gcloud secrets versions add charitychain-prod-secret-key

# 3. Redeploy to pick up new version (zero-downtime)
gcloud run services update charitychain-backend-prod \
  --region=us-central1 \
  --update-secrets=SECRET_KEY=charitychain-prod-secret-key:latest

# 4. Disable old version after verification
gcloud secrets versions disable charitychain-prod-secret-key \
  --version=PREVIOUS_VERSION_NUMBER
```

> [!WARNING]
> Rotating JWT_SECRET_KEY will immediately invalidate all active user sessions. Send advance notice to users.

---

## RB-04: Investigating a Slow API Response

**Trigger**: P95 response time > 2 seconds.

**Steps:**
1. Check the `app-health` Grafana dashboard for latency percentiles
2. Check slow query log in Cloud SQL:
   ```sql
   SELECT query, calls, mean_exec_time, total_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 20;
   ```
3. Check if rate-limiting is active (Prometheus metric: `charitychain_rate_limit_hits_total`)
4. Check Cloud Run instance count — may need to increase min-instances
5. If a specific endpoint is slow: Check SQLAlchemy N+1 queries via Sentry performance

---

## RB-05: Restarting the Notification Outbox Worker

**Trigger**: Outbox queue backing up; notifications not being delivered.

**Steps:**
```bash
# Check outbox queue depth
psql -c "SELECT COUNT(*), status FROM notification_outbox GROUP BY status;"

# Check dead-letter queue
psql -c "SELECT COUNT(*) FROM notification_outbox WHERE status = 'DEAD_LETTER';"

# If using Cloud Run Job for outbox:
gcloud run jobs execute charitychain-outbox-worker-prod --region=us-central1

# If using CLI worker locally:
python -m backend.app.cli run-notification-worker --once --batch-size=50
```

---

## RB-06: Handling a Failed PayPal Webhook

**Trigger**: PayPal webhook not reflected in donation status.

**Steps:**
1. Check webhook events in database:
   ```sql
   SELECT event_id, event_type, processing_status, created_at
   FROM webhook_events
   WHERE source = 'PAYPAL'
   ORDER BY created_at DESC
   LIMIT 20;
   ```
2. For failed events, check webhook processing status
3. If signature verification failure: Verify `PAYPAL_WEBHOOK_ID` matches PayPal dashboard
4. Manually trigger reprocessing for a specific event ID via admin API
5. Check PayPal Developer Dashboard for webhook delivery logs

---

## RB-07: Database Connection Pool Exhaustion

**Trigger**: 503 errors with "connection refused" or "pool exhausted" in logs.

**Steps:**
1. Check active connections:
   ```sql
   SELECT count(*), state FROM pg_stat_activity
   WHERE datname = 'charitychain' GROUP BY state;
   ```
2. Terminate idle connections:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'charitychain'
     AND state = 'idle'
     AND query_start < NOW() - INTERVAL '10 minutes';
   ```
3. If Cloud Run has too many instances: Temporarily reduce max-instances
4. Review SQLAlchemy pool settings: `pool_size`, `max_overflow`, `pool_timeout`

---

## RB-08: Scaling Up for Traffic Spike

**Trigger**: High response latency or Grafana alert for increased traffic.

**Steps:**
```bash
# Increase maximum instances for backend
gcloud run services update charitychain-backend-prod \
  --region=us-central1 \
  --max-instances=50 \
  --min-instances=5

# Increase maximum instances for frontend
gcloud run services update charitychain-frontend-prod \
  --region=us-central1 \
  --max-instances=20 \
  --min-instances=3
```

> Cloud Run scales automatically up to max-instances. Increase max-instances, not min-instances, to handle spikes without idle costs.

---

## RB-09: Restoring Evidence Files

See `docs/PHASE_14_DISASTER_RECOVERY_DRILLS.md` DR-04.

---

## RB-10: Creating an Admin User via CLI

**Use case**: Initial admin setup or locked-out admin recovery.

```bash
# From inside a running backend container or local environment
python -m backend.app.cli create-admin \
  --email admin@charitychain.dev \
  --password "STRONG_TEMPORARY_PASSWORD"
```

---

## RB-11: Verifying Security Audit Chain Integrity

```bash
curl -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "https://api.charitychain.dev/api/v1/admin/security/audit-logs/verify?limit=1000"
```

Expected response: `{"chain_valid": true, "events_checked": N}`

If chain is invalid: Stop all write operations, preserve disk state, engage security team.

---

## RB-12: Certificate Verification Failure Investigation

**Scenario**: Impact certificate fails public verification.

**Steps:**
1. Check certificate status:
   ```sql
   SELECT id, status, signing_key_id, issued_at, revoked_at
   FROM impact_certificates WHERE id = 'CERTIFICATE_ID';
   ```
2. Check JWKS endpoint: `curl https://api.charitychain.dev/.well-known/jwks.json`
3. Verify signing key is not retired: Check `issuer_signing_keys.status`
4. If key was rotated: Ensure JWKS endpoint returns both old and new keys
5. Re-run certificate verification manually:
   ```bash
   python -m backend.app.cli verify-certificate --certificate-id CERT_ID
   ```

---

## RB-13: Responding to a Security Audit Alert

**Trigger**: Grafana `security-telemetry` dashboard alert.

**Immediate steps:**
1. Identify the alert type (brute force, rate limit, auth failure spike)
2. Check security audit trail:
   ```bash
   curl -H "Authorization: Bearer ${ADMIN_TOKEN}" \
     "https://api.charitychain.dev/api/v1/admin/security/audit-logs?limit=50&event_type=AUTH_FAILURE"
   ```
3. If brute force on specific account: Account is auto-locked after 5 failures (15 min)
4. If distributed attack: Block IPs at nginx/load-balancer level
5. If credential stuffing: Enable enhanced monitoring, notify security team
6. Document in incident log regardless of severity

---

## RB-14: Running Database Migrations

See `docs/PHASE_14_DATABASE_MIGRATION_STRATEGY.md`.

---

## RB-15: Verifying Impact Ledger Chain

```bash
curl -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "https://api.charitychain.dev/api/v1/impact-ledger/verify"
```

Expected: `{"chain_valid": true, "events_verified": N, "latest_hash": "..."}`

---

## RB-16: Checking Financial Reconciliation Status

```bash
curl -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "https://api.charitychain.dev/api/v1/reconciliation/runs?status=DISCREPANCY_FOUND"
```

If discrepancies found: Review `reconciliation_items` for specific mismatched amounts. Never auto-resolve financial discrepancies — always requires human review.

---

## RB-17: Enabling Live PayPal Transactions

> [!CAUTION]
> Live PayPal requires explicit authorization from the Product and Finance teams.
> This action enables real financial transactions. Never enable without approval.

**Prerequisites:**
- PayPal live account approved and verified
- Live `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET` configured in Secret Manager
- Live `PAYPAL_WEBHOOK_ID` configured
- Financial team sign-off documented

**Steps:**
```bash
# 1. Update Secret Manager with live credentials (done separately)

# 2. Enable feature flag
gcloud run services update charitychain-backend-prod \
  --region=us-central1 \
  --update-env-vars="PAYPAL_ENVIRONMENT=live,FEATURE_PAYPAL_LIVE_ENABLED=true"

# 3. Verify health
curl https://api.charitychain.dev/health

# 4. Run a test donation with a small real amount to confirm end-to-end flow

# 5. Monitor Sentry and Grafana financial-ops dashboard for 30 minutes
```

