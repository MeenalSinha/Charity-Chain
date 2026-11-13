# CharityChain — Phase 13 Security Incident Response Plan (IRP)
**Document Version:** 1.0.0  
**Effective Date:** November 13, 2026  
**Audience:** Security Incident Response Team (SIRT), DevOps, Platform Leadership  

---

## 1. Incident Severity Classification

| Severity Level | Definition | Response SLA | Escalation Target |
|---|---|:---:|---|
| **P0 — Critical** | System compromise, private key leak, financial ledger corruption, active data exfiltration | **Immediate (< 15 mins)** | Incident Commander, Lead Architect, Legal Counsel |
| **P1 — High** | Sustained DDoS, brute-force bypass, dual-key review failure, webhook forgery | **< 1 hour** | SIRT Lead, Senior Backend Engineer |
| **P2 — Medium** | Isolated credential stuffing, rate-limiting degradation, non-critical service disruption | **< 4 hours** | On-Call Security Engineer |
| **P3 — Low** | Minor vulnerability scan finding, transient outbox error, isolated certificate warning | **< 24 hours** | Platform Engineering Team |

---

## 2. Incident Response Lifecycle

```
[ 1. DETECTION ] ---> [ 2. CONTAINMENT ] ---> [ 3. ERADICATION ] ---> [ 4. RECOVERY ] ---> [ 5. POST-MORTEM ]
- Sentry Alerts       - Revoke Sessions       - Patch Vulnerability   - Restore Backups     - Root Cause Analysis
- Prometheus Alarms   - Rotate Keys           - Ban Malicious IPs     - Verify Chain Hash   - Preventive Actions
- Audit Tamper Flag   - Isolate Services      - Flush Redis Cache     - Canary Traffic      - Publish PIR
```

---

## 3. Step-by-Step Incident Containment Runbooks

### 3.1 Runbook 1: Cryptographic Key Compromise (Ed25519 or JWT Secret)
1. **Immediate Revocation**: Mark the compromised key ID as `REVOKED` in the database.
2. **Key Generation**: Generate a fresh Ed25519 keypair via `python -m backend.app.cli generate-issuer-key`.
3. **Session Invalidation**: Execute `AuthService.revoke_all_sessions()` to force global re-authentication.
4. **JWKS Rotation**: Deploy new public keys to `GET /.well-known/jwks.json`.
5. **Certificate Audit**: Review all certificates signed during the compromise window for fraudulent issuance.

### 3.2 Runbook 2: Credential Stuffing / Brute-Force Attack
1. **Identify Source**: Extract top offending IP ranges from Prometheus `charitychain_rate_limit_hits_total{tier="auth"}`.
2. **Edge Throttling**: Configure temporary edge firewall blocks on offending IP blocks.
3. **User Lockout Verification**: Verify that accounts with 5 failed attempts are locked in database.
4. **Password Reset Broadcast**: Trigger forced password resets for targeted accounts.

### 3.3 Runbook 3: Security Audit Hash Chain Tampering
1. **Trigger Condition**: `SecurityAuditService.verify_chain_integrity()` reports `is_valid: False`.
2. **Isolate Audit Log**: Immediately restrict write permissions to `security_audit_events` to read-only.
3. **Identify Point of Breakage**: Retrieve `broken_at_event_id` and query PostgreSQL WAL logs to determine the exact timestamp, database user, and IP that altered the record.
4. **Evidence Preservation**: Take forensic snapshots of database tables and server disk state.
5. **Report to Leadership**: Notify Platform Leadership and initiate internal security review.

### 3.4 Runbook 4: Double-Entry Ledger Trial Balance Imbalance
1. **Trigger Condition**: General ledger reconciliation detects $\sum \text{Debits} \ne \sum \text{Credits}$.
2. **Freeze Financial Operations**: Temporarily suspend outbound disbursements via `settings.PAYOUT_PROVIDER = "mock"`.
3. **Audit Journal Entries**: Run query to isolate unposted or unbalanced `journal_entries`.
4. **Correction via Adjustment**: Author adjusting journal entry with dual-auditor approval (never update historical ledger rows directly).
5. **Resume Operations**: Re-verify trial balance equals zero and unfreeze disbursements.

---

## 4. Post-Incident Review (PIR) Template
Every P0 and P1 incident requires a blameless post-mortem within 48 hours covering:
- **Incident Summary**: What happened and what was the customer/financial impact.
- **Root Cause Analysis (5 Whys)**: The technical and systemic causes.
- **Timeline**: Chronological sequence from trigger to detection, containment, and resolution.
- **Action Items**: Preventative measures with owners and due dates.

