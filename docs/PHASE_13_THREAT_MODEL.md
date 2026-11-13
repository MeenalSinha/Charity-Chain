# CharityChain — Phase 13 Threat Model
**Classification:** Internal Security Architecture Document  
**Version:** 1.0.0  
**Methodology:** STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) & DREAD Scoring  
**Date:** November 13, 2026  

---

## 1. Executive Summary & Scope

CharityChain processes donor financial transactions, distributes donor funds to non-governmental organizations (NGOs), archives sensitive organizational legal documents, evaluates multimedia milestone evidence via automated AI pipelines, and issues cryptographically signed Proof-of-Impact certificates.

This threat model identifies high-value assets, diagrams system trust boundaries, catalogues potential attack vectors, evaluates risks using DREAD scoring, and defines technical and operational countermeasures.

---

## 2. Asset Classification

| Asset Category | Data Elements | Confidentiality | Integrity | Availability | Risk Classification |
|---|---|:---:|:---:|:---:|:---:|
| **Cryptographic Secrets** | Ed25519 Certificate Signing Keys, JWT Secrets, Fernet AES-256 Webhook Keys, Argon2id Hashes | **CRITICAL** | **CRITICAL** | **HIGH** | **Tier 1 (Catastrophic)** |
| **Financial Ledger & Payment Data** | Double-entry general ledger, PayPal order/capture/refund tokens, bank disbursement details | **HIGH** | **CRITICAL** | **HIGH** | **Tier 1 (Catastrophic)** |
| **Identity & Authentication** | User emails, session refresh tokens, password reset tokens, MFA credentials | **HIGH** | **HIGH** | **HIGH** | **Tier 2 (High Risk)** |
| **NGO Legal & Evidence Records** | Tax filings, government registration PDFs, raw milestone media with EXIF, reviewer notes | **HIGH** | **HIGH** | **MEDIUM** | **Tier 2 (High Risk)** |
| **Security Audit Records** | Tamper-evident hash-chained audit events, administrative logs | **MEDIUM** | **CRITICAL** | **HIGH** | **Tier 2 (High Risk)** |
| **Public Transparency Data** | Project descriptions, milestone progress, public metrics, verified certificates | **LOW** | **HIGH** | **HIGH** | **Tier 3 (Operational)** |

---

## 3. System Trust Boundaries & Data Flow

```
[ Unauthenticated Client / Browser / External Donor ]
                   |
     (TLS 1.3 / HTTPS / HSTS / CSP)
                   v
       [ Next.js 16 BFF Edge (Port 3000) ]
       * HttpOnly Cookie Session Management
       * Header Sanitization & CSRF Defenses
                   |
     (Internal Network / localhost:8000)
                   v
       [ FastAPI Backend (Port 8000) ]
       * Sliding Window Rate Limiter (Redis)
       * RBAC & Organization Tenant Scoping
       * Zero-Trust Input Hardening & Sanitization
       * Sentry Error Redaction (Zero PII)
          /         |          \          \
         v          v           v          v
   [ PostgreSQL ] [ Redis ] [ Local/Cloud ] [ External Services ]
   * Minor-unit   * Rate-    Storage        * PayPal REST API
     accounting     limits   * Magic byte   * SMTP / TLS
   * Double-entry * Sessions   validation   * Sentry Error Ingest
     ledger                  * Path-        * Prometheus Scrape
   * Hash-chain                traversal      (Authenticated)
     audit logs                defense
```

### Trust Boundary Definitions:
1. **Boundary A (Public Internet to BFF / Next.js)**: Untrusted boundary. High risk of DDoS, credential stuffing, XSS, and CSRF.
2. **Boundary B (BFF to FastAPI Backend)**: Authenticated server-to-server or secure proxy boundary. Validates correlation IDs, bearer tokens, and tenant scoping.
3. **Boundary C (Backend to Persistence - PostgreSQL / Redis)**: Trusted internal boundary. Enforces parameterized queries, encrypted connections, and connection pooling.
4. **Boundary D (Backend to Storage Providers)**: File validation boundary. Enforces binary magic byte inspection, SHA-256 hashing, and anti-path-traversal sanitization.
5. **Boundary E (Backend to Third-Party Egress - PayPal, Webhooks, Sentry)**: Egress boundary. Enforces anti-SSRF IP filtering (RFC 1918 blocking), HMAC signature verification, and recursive PII scrubbing.

---

## 4. STRIDE Threat Analysis

### 4.1 Spoofing (Identity Deception)
- **Threat S1: Credential Stuffing & Brute Force Login**
  - *Risk*: Attackers test credential dumps against `/api/v1/auth/login`.
  - *Mitigations*: Argon2id password hashing, constant-time verification, automatic 15-minute account lockout after 5 consecutive failures, and Redis sliding-window rate limiting (5 requests/minute per IP).
- **Threat S2: Impersonation via Webhook Forgery**
  - *Risk*: Malicious actors send fake PayPal payout or capture webhooks.
  - *Mitigations*: Cryptographic webhook signature verification via PayPal SDK/API before transaction state changes; replay rejection via unique event ID deduplication.

### 4.2 Tampering (Unauthorized Modification)
- **Threat T1: General Ledger & Financial Record Alteration**
  - *Risk*: Direct or SQLi modification of balance numbers.
  - *Mitigations*: Strict double-entry accounting with balanced debits and credits ($\sum D == \sum C$); PostgreSQL constraints; immutable append-only transaction design.
- **Threat T2: Impact Certificate & Proof-of-Impact Forgery**
  - *Risk*: Counterfeit completion certificates issued for fraudulent milestones.
  - *Mitigations*: Asymmetric Ed25519 cryptographic signing with private key stored in secure environment; RFC 8785 canonical JSON hashing; 9-gate eligibility service requiring dual-key human approval.
- **Threat T3: Security Audit Log Tampering**
  - *Risk*: Rogue administrator modifies or deletes audit logs to cover tracks.
  - *Mitigations*: Centralized SHA-256 hash-chaining where each entry incorporates the preceding entry's hash; append-only table permissions; cryptographic chain verification endpoints.

### 4.3 Repudiation (Denial of Actions)
- **Threat R1: Reviewer Disputes Approval of Fraudulent Evidence**
  - *Risk*: Human reviewer claims their account was not used to approve milestone evidence.
  - *Mitigations*: Non-repudiation audit trails linking the authenticated reviewer's user ID, IP address, timestamp, dual-key peer ID, and cryptographic hash of the reviewed evidence packet.

### 4.4 Information Disclosure (Data Leakage)
- **Threat I1: Secret and PII Exposure in Telemetry/Logs**
  - *Risk*: API keys, JWT tokens, donor emails, or payment IDs logged to Sentry or Prometheus.
  - *Mitigations*: Recursive redaction filter stripping sensitive keys (`password`, `token`, `secret`, `authorization`, `cookie`, `cvv`) prior to log dispatch or Sentry transmission.
- **Threat I2: Geographic Beneficiary Deanonymization**
  - *Risk*: Precise GPS coordinates in evidence disclose locations of vulnerable aid recipients.
  - *Mitigations*: Public API coordinate obfuscation with ~11.1 km centroid coarsening for location-sensitive projects (`is_location_sensitive=True`).

### 4.5 Denial of Service (Availability Disruption)
- **Threat D1: AI Verification Pipeline Starvation**
  - *Risk*: Malicious users flood `/api/v1/evidence/upload` with gigabytes of nested files to exhaust server CPU and memory.
  - *Mitigations*: Strict 10MB file size ceiling, binary magic byte validation, Redis sliding-window rate limiting (10 requests/minute per organization), and asynchronous background worker processing.
- **Threat D2: Database Connection Exhaustion via Slowloris**
  - *Risk*: Slow HTTP requests consume FastAPI connection pool.
  - *Mitigations*: Reverse proxy read/write timeouts (10s), connection pool limits (SQLAlchemy max_overflow=10), and client-side abort controllers.

### 4.6 Elevation of Privilege (Access Escalation)
- **Threat E1: Insecure Direct Object References (IDOR) on Org Documents**
  - *Risk*: User of Organization A accesses confidential tax documents of Organization B.
  - *Mitigations*: Strict tenant isolation enforcement in service layers (`organization_id` matching authenticated user membership); reviewer affiliation checks preventing conflicts of interest.
- **Threat E2: Platform Role Escalation**
  - *Risk*: Normal donor updates their user record to `PLATFORM_ADMIN`.
  - *Mitigations*: Role changes restricted to CLI bootstrapping or platform admin endpoints guarded by `require_platform_role(PlatformRole.PLATFORM_ADMIN)`.

---

## 5. DREAD Risk Assessment Matrix

| Threat ID | Threat Description | Damage (1-10) | Reproducibility (1-10) | Exploitability (1-10) | Affected Users (1-10) | Discoverability (1-10) | DREAD Score (Sum/5) | Risk Level |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1** | Credential Stuffing & Brute Force | 7 | 8 | 8 | 6 | 8 | **7.4** | **HIGH** |
| **S2** | Inbound Webhook Forgery | 9 | 5 | 4 | 8 | 5 | **6.2** | **HIGH** |
| **T1** | Financial Ledger Manipulation | 10 | 3 | 3 | 10 | 4 | **6.0** | **HIGH** |
| **T2** | Certificate Signature Forgery | 9 | 2 | 2 | 10 | 3 | **5.2** | **MEDIUM** |
| **T3** | Audit Trail Tampering | 8 | 4 | 3 | 9 | 4 | **5.6** | **MEDIUM** |
| **I1** | Telemetry PII / Credential Leakage | 8 | 7 | 6 | 9 | 7 | **7.4** | **HIGH** |
| **I2** | Beneficiary Location De-anonymization | 8 | 6 | 5 | 5 | 6 | **6.0** | **HIGH** |
| **D1** | Evidence Pipeline Resource Exhaustion | 7 | 8 | 7 | 8 | 7 | **7.4** | **HIGH** |
| **E1** | Multi-Tenant IDOR Document Access | 8 | 7 | 6 | 7 | 7 | **7.0** | **HIGH** |

---

## 6. Security Countermeasure Implementation Plan

1. **Sliding-Window Rate Limiter**: Redis-backed token bucket/sliding log limiting burst and sustained abuse.
2. **Tamper-Evident Security Audit Service**: SHA-256 hash-chained event log covering all authentication and administrative actions.
3. **Observability & Error Hardening**: Sentry SDK with recursive PII scrubber and Prometheus `/metrics` scraping.
4. **OWASP Hardening**: HTTP Strict Transport Security, Content Security Policy, Permissions Policy, and non-root container profiles.
5. **SOC 2 Readiness**: Operational controls for access review, vulnerability management, and incident response.

