# Phase 14 — Disaster Recovery Drills

**Version:** 1.0  
**Date:** 2026-11-14  
**Classification:** Operations — Confidential

> [!CAUTION]
> Never execute production DR drills without change-management approval and a maintenance window.
> All drill results must be recorded in the Incident Management System.

---

## 1. DR Drill Schedule

| Drill | Frequency | Participants | Last Completed |
|:---|:---|:---|:---:|
| Backup restore verification | Weekly (automated) | Automated | — |
| Full database failover | Quarterly | DBA + On-call | — |
| Regional failover | Semi-annually | Full engineering team | — |
| Signing key restore | Annually | Security lead | — |
| PayPal outage simulation | Quarterly | Payments team | — |

---

## 2. Drill Procedures

---

### DR-01: PostgreSQL Point-in-Time Recovery (PITR)

**Scenario**: Accidental mass deletion of donation records.  
**RTO target**: 1 hour  
**RPO target**: 5 minutes  

**Procedure:**
1. Identify the target recovery timestamp (time just before data loss)
2. Create restore from Cloud SQL PITR:
   ```bash
   gcloud sql instances clone charitychain-prod-db charitychain-prod-db-restore \
     --point-in-time "2026-11-14T02:00:00.000Z"
   ```
3. Verify restore: Connect to `charitychain-prod-db-restore` and count records
4. Run integrity checks:
   ```sql
   SELECT COUNT(*) FROM donations WHERE status = 'CAPTURED';
   SELECT SUM(amount_minor_units) FROM payment_captures;
   ```
5. Update application DATABASE_URL to point to restored instance
6. Run Alembic migration check: `alembic current`
7. Verify financial reconciliation reports
8. Promote restored instance: Rename original, rename restore to production name
9. Reconnect application to production database

**Success criteria**: Financial records present, ledger chain intact, certificates verifiable.

---

### DR-02: Full pg_dump Restore (Encrypted Backup)

**Scenario**: Complete data centre failure with Cloud SQL unavailable.  
**RTO target**: 2 hours  
**RPO target**: 24 hours (last nightly backup)  

**Procedure:**
1. Provision new PostgreSQL 16 instance (Cloud SQL, GKE, or VM)
2. Download latest encrypted backup:
   ```bash
   LATEST=$(gsutil ls gs://charitychain-backups-prod-PROJECT/postgres/ | sort | tail -1)
   gsutil cp "${LATEST}" /tmp/restore.sql.gz.gpg
   ```
3. Decrypt:
   ```bash
   echo "${BACKUP_ENCRYPTION_KEY}" | gpg --batch --passphrase-fd 0 \
     --decrypt --output /tmp/restore.sql.gz /tmp/restore.sql.gz.gpg
   ```
4. Restore:
   ```bash
   gunzip -c /tmp/restore.sql.gz | psql --host=NEW_HOST --username=charity_user charitychain
   ```
5. Run integrity checks (see DR-01 step 4)
6. Update application connection strings via Secret Manager
7. Restart backend services
8. Verify health endpoint: `curl https://api.charitychain.dev/health`

**Success criteria**: Application serving requests within 2 hours.

---

### DR-03: Redis Failure Recovery

**Scenario**: Redis Memorystore instance unavailable.  
**Expected behavior**: Rate limiter falls back to in-memory, tokens re-issued.  
**RTO target**: 15 minutes (automatic failover for STANDARD_HA tier)

**Procedure:**
1. Confirm Memorystore auto-failover has completed (STANDARD_HA has built-in HA)
2. If manual intervention needed:
   ```bash
   gcloud redis instances failover charitychain-prod-redis \
     --region=us-central1 \
     --data-protection-mode=limited-data-loss
   ```
3. Verify Redis connectivity from backend:
   ```bash
   redis-cli -h REDIS_HOST -p 6379 -a "${REDIS_PASSWORD}" ping
   ```
4. Monitor rate-limiting metrics — if in-memory fallback active, alert engineering
5. Confirm active session tokens are still valid (JWTs do not require Redis)

**Warning**: Active rate-limit windows will be lost on Redis restart. This briefly allows higher request rates than configured.

---

### DR-04: Evidence File Recovery (GCS Object Restoration)

**Scenario**: Evidence files accidentally deleted from GCS bucket.  
**RTO target**: 30 minutes  

**Procedure:**
1. Identify affected files from database:
   ```sql
   SELECT storage_key, sha256_hash FROM evidence_files WHERE is_archived = false;
   ```
2. For each missing file, restore from GCS versioning:
   ```bash
   gsutil cp \
     gs://charitychain-evidence-prod/evidence/FILE_KEY#VERSION_ID \
     gs://charitychain-evidence-prod/evidence/FILE_KEY
   ```
3. Verify SHA-256 checksum:
   ```bash
   gsutil hash -h gs://charitychain-evidence-prod/evidence/FILE_KEY
   ```
4. Run database integrity check using the evidence verification API:
   ```bash
   curl -H "Authorization: Bearer ${ADMIN_TOKEN}" \
     https://api.charitychain.dev/api/v1/evidence/EVIDENCE_ID/verify
   ```

---

### DR-05: Signing Key Recovery

**Scenario**: Primary Ed25519 signing key lost or compromised.  
**RTO target**: 2 hours  
**Impact**: New certificates cannot be issued until key is restored.

**Procedure:**
1. Immediately disable certificate issuance (set `FEATURE_CERTIFICATES_ENABLED=false` via Secret Manager)
2. Retrieve key from backup source (Secret Manager secondary project or offline HSM)
3. Load key into production Secret Manager:
   ```bash
   gcloud secrets create charitychain-prod-signing-key-v2 \
     --data-file=/path/to/recovered-key.pem
   ```
4. Update backend to use new key version:
   ```bash
   gcloud run services update charitychain-backend-prod \
     --update-secrets=SIGNING_KEY_PATH=charitychain-prod-signing-key-v2:latest \
     --region=us-central1
   ```
5. If key is compromised (not just lost): Generate new key pair, update JWKS endpoint, mark old certificates with `KEY_COMPROMISED` status flag
6. Re-enable certificate issuance
7. Notify affected NGOs and donors via email

---

### DR-06: Backend Service Outage (Cloud Run)

**Scenario**: Backend Cloud Run service not responding.  
**RTO target**: 5 minutes  

**Procedure:**
1. Check Cloud Run status:
   ```bash
   gcloud run services describe charitychain-backend-prod --region=us-central1
   gcloud logging read "resource.type=cloud_run_revision" --limit=50 --format=json
   ```
2. Rollback to previous revision if new deployment caused issue:
   ```bash
   PREV_REV=$(gcloud run revisions list \
     --service=charitychain-backend-prod \
     --region=us-central1 \
     --format="value(name)" \
     --sort-by="~creationTimestamp" \
     --limit=2 | tail -1)
   gcloud run services update-traffic charitychain-backend-prod \
     --region=us-central1 \
     --to-revisions="${PREV_REV}=100"
   ```
3. Verify rollback success: `curl https://api.charitychain.dev/health`
4. Investigate root cause via Cloud Logging and Sentry

---

### DR-07: Database Connection Pool Exhaustion

**Scenario**: Backend cannot connect to Cloud SQL due to connection exhaustion.  
**RTO target**: 10 minutes  

**Procedure:**
1. Check connection count:
   ```sql
   SELECT COUNT(*) FROM pg_stat_activity WHERE datname = 'charitychain';
   ```
2. Identify long-running queries:
   ```sql
   SELECT pid, now() - query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active' AND duration > interval '5 minutes'
   ORDER BY duration DESC;
   ```
3. Terminate idle connections:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'charitychain' AND state = 'idle' AND duration > interval '10 minutes';
   ```
4. If Cloud Run instances overloaded: Temporarily scale down min-instances
5. Long-term: Review connection pool settings in SQLAlchemy (pool_size, max_overflow)

---

### DR-08: PayPal Gateway Outage

**Scenario**: PayPal API unavailable — donations cannot be processed.  
**RTO target**: N/A (external dependency)  

**Procedure:**
1. Monitor PayPal status page: https://www.paypal-status.com/
2. Display maintenance banner on frontend donation flow
3. Queue donation intents in database with `GATEWAY_UNAVAILABLE` status
4. When PayPal recovers: Resume processing queued intents via outbox worker
5. Send email notification to donors whose donations are queued

**Note**: CharityChain stores all payment data server-side before gateway submission. No donations are lost during gateway outages.

---

### DR-09: Security Breach — Credential Compromise

**Scenario**: JWT_SECRET_KEY or database credentials compromised.  
**Severity**: CRITICAL — Execute immediately  

**Procedure:**
1. **Rotate JWT_SECRET_KEY immediately** (all existing tokens become invalid):
   ```bash
   NEW_KEY=$(python -c "import secrets; print(secrets.token_hex(64))")
   gcloud secrets versions add charitychain-prod-jwt-key --data="${NEW_KEY}"
   # Redeploy to pick up new secret version
   gcloud run services update charitychain-backend-prod --region=us-central1 \
     --update-secrets=JWT_SECRET_KEY=charitychain-prod-jwt-key:latest
   ```
2. **Revoke all active sessions** in database:
   ```sql
   UPDATE user_sessions SET status = 'REVOKED', revoked_at = NOW()
   WHERE status = 'ACTIVE';
   ```
3. If database credentials compromised:
   ```bash
   # Create new DB user
   gcloud sql users set-password charity_user \
     --instance=charitychain-prod-db \
     --password="${NEW_DB_PASSWORD}"
   # Update secret
   echo -n "${NEW_DB_PASSWORD}" | gcloud secrets versions add charitychain-prod-db-password
   # Redeploy
   gcloud run services update charitychain-backend-prod --region=us-central1 \
     --update-secrets=DATABASE_URL=charitychain-prod-db-url:latest
   ```
4. Notify security team and begin forensic investigation
5. File incident report within 24 hours

---

## 3. Drill Recording Template

```
DRILL RECORD
============
Drill ID: DR-{NUMBER}
Date: YYYY-MM-DD
Participants: 
Environment: staging / production
Trigger: scheduled / simulated

Procedure Followed: steps executed, any deviations noted
Time Started: HH:MM UTC
Time Completed: HH:MM UTC
Actual RTO: X minutes
Actual RPO: Y minutes

Data Loss: none / X minutes / X hours
Issues Encountered: description of any problems
Improvements Identified: list of follow-up actions

Signed: participant names
```

---

## 4. DR Contact Directory

| Role | Responsibility |
|:---|:---|
| On-call engineer | First responder, all DR scenarios |
| Database administrator | DR-01, DR-02, DR-07 |
| Security lead | DR-05, DR-09 |
| Payments specialist | DR-08 |
| Operations manager | Authorizes production DR execution |

