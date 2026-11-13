# Phase 14 — Backup Strategy

**Version:** 1.0  
**Date:** 2026-11-14  
**Classification:** Operations — Internal

---

## 1. Overview

CharityChain handles financial transactions, cryptographic impact certificates, evidence files, and audit trails. Data loss in any of these categories could have irreversible consequences for NGOs, donors, and legal compliance.

This document defines the backup strategy for all production data, including backup frequency, retention, encryption, verification, and recovery procedures.

---

## 2. Data Assets and RPO/RTO Targets

| Data Asset | Backup Method | RPO | RTO | Retention |
|:---|:---|:---:|:---:|:---:|
| PostgreSQL database | pg_dump + Cloud SQL PITR | 5 min | 1 hour | 30 days (backups) + 7 years (archives) |
| Evidence files (GCS) | GCS versioning + cross-region replication | Near-zero | 30 min | 7 years |
| Signing keys (Ed25519) | Secret Manager + offline backup | Near-zero | 2 hours | Indefinite |
| Grafana state | docker volume backup weekly | 7 days | 2 hours | 30 days |
| Terraform state | GCS (versioned bucket) | Near-zero | 15 min | Indefinite |

**RPO** = Recovery Point Objective (maximum acceptable data loss)  
**RTO** = Recovery Time Objective (maximum acceptable downtime)

---

## 3. PostgreSQL Backup Strategy

### 3.1 Automated Backups

**Cloud SQL automated backups** (production):
- Daily automated backups retained for 30 days
- Point-in-Time Recovery (PITR) enabled — recover to any second in the last 7 days
- Backups stored in a different GCP region from the primary instance
- Backup window: 03:00–04:00 UTC (lowest traffic period)

**pg_dump backups** (all environments):
- Script: `scripts/backup/backup_postgres.sh`
- Schedule: Nightly at 02:00 UTC (before Cloud SQL backup window)
- Format: Plain SQL, gzip-compressed, AES-256-GPG encrypted
- Storage: Dedicated GCS backup bucket (`charitychain-backups-prod-{PROJECT_ID}`)
- Retention: 90 days hot, then archived to Nearline/Coldline storage

### 3.2 Backup Verification

Automated verification runs weekly:
- Script: `scripts/backup/verify_backup.sh`
- Downloads a randomly selected backup from GCS
- Decrypts and restores to an isolated verification PostgreSQL instance
- Runs integrity checks (table count, foreign key validation, financial reconciliation)
- Sends verification result to monitoring

**A backup that has not been verified is not a backup.**

### 3.3 Backup Encryption

All pg_dump backups are encrypted with:
- Algorithm: GPG symmetric AES-256
- Passphrase: Stored in GCP Secret Manager (`charitychain-prod-backup-encryption-key`)
- Key rotation: Every 90 days (new backups re-encrypted; old backups retained under old key)
- Offline copy: Passphrase stored in offline hardware key management system

### 3.4 Backup Storage Security

| Property | Setting |
|:---|:---|
| Bucket access | Private, no public access |
| IAM | Only backup service account has write; ops team has read |
| Versioning | Enabled |
| Object lifecycle | 90 days → Nearline; 365 days → Coldline; 7 years → delete |
| Encryption | Google-managed CMK (AES-256) at rest |
| Cross-region replication | Enabled (dual-region bucket) |

---

## 4. Evidence File Backup Strategy

Evidence files are stored in GCS with:
- **Versioning enabled**: Every version of every evidence file retained
- **Object retention lock**: 7-year legal hold on files linked to issued certificates
- **Cross-region replication**: Automatic replication to secondary bucket
- **Integrity checksums**: SHA-256 checksums stored in database, verified on download

---

## 5. Signing Key Backup Strategy

Ed25519 signing keys for impact certificates require special handling:

1. **Primary**: GCP Secret Manager (`charitychain-prod-signing-key`)
2. **Secondary**: Encrypted copy in a separate GCP project's Secret Manager
3. **Offline**: Air-gapped backup on hardware security token (HSM or FIDO2 key)
4. **Key retirement**: Retired keys are retained for 10 years (certificate verification requires original key)

---

## 6. Backup Schedule

| Time (UTC) | Task |
|:---|:---|
| 02:00 daily | pg_dump to GCS (all environments) |
| 03:00 daily | Cloud SQL automated backup |
| Sunday 04:00 | Backup verification (restore and integrity check) |
| 1st of month 05:00 | Full database export (CSV format, for compliance reporting) |
| Quarterly | Signing key backup rotation |

---

## 7. Operational Procedures

### 7.1 Trigger Manual Backup

```bash
# Production backup
BACKUP_BUCKET=charitychain-backups-prod-PROJECT_ID \
POSTGRES_HOST=... \
POSTGRES_USER=charity_user \
POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret=charitychain-prod-db-password) \
POSTGRES_DB=charitychain \
BACKUP_ENCRYPTION_KEY=$(gcloud secrets versions access latest --secret=charitychain-prod-backup-encryption-key) \
./scripts/backup/backup_postgres.sh
```

### 7.2 Verify Latest Backup

```bash
# Find latest backup
LATEST=$(gsutil ls gs://charitychain-backups-prod-PROJECT_ID/postgres/ | sort | tail -1)

# Verify
BACKUP_ENCRYPTION_KEY=$(gcloud secrets versions access latest --secret=charitychain-prod-backup-encryption-key) \
VERIFY_DB_HOST=... \
VERIFY_DB_USER=... \
VERIFY_DB_PASSWORD=... \
./scripts/backup/verify_backup.sh "${LATEST}"
```

### 7.3 List Available Backups

```bash
gsutil ls -l gs://charitychain-backups-prod-PROJECT_ID/postgres/ | sort -k2
```

---

## 8. Compliance

| Requirement | Implementation |
|:---|:---|
| SOC 2 CC9.1 — Business continuity | Automated nightly backups with PITR |
| SOC 2 CC7.5 — Recovery testing | Weekly verified restore |
| Data retention (7 years) | GCS lifecycle rules, Cloud SQL PITR |
| Encryption at rest | AES-256 (GCS) + GPG (pg_dump) |
| Encryption in transit | TLS 1.2+ on all GCS/Cloud SQL connections |
| Access control | IAM — backup SA has write-only; ops has read |

