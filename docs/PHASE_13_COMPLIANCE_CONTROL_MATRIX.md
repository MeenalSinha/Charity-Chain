# CharityChain — Phase 13 SOC 2 Compliance Readiness Control Matrix
**Document Version:** 1.0.0  
**Scope:** SOC 2 Type II Readiness Framework (Security, Availability, Processing Integrity, Confidentiality, Privacy)  
**Notice:** *This document represents internal readiness preparation and control mapping. CharityChain does NOT claim SOC 2 certification or attestation until a formal third-party CPA audit has concluded.*  

---

## 1. Executive Summary

This matrix aligns CharityChain's operational policies, technical implementations, cryptographic controls, and monitoring systems with the **AICPA 2017 Trust Services Criteria (TSC)**.

---

## 2. Common Criteria (Security) Control Mapping

| Control ID | TSC Reference | Control Description | CharityChain Implementation | Evidence / Verification Pointer |
|---|---|---|---|---|
| **CC1.1** | COSO Principle 1 | Tone at the top and code of conduct | Documented Security & Ethics Policies | `docs/governance/` |
| **CC2.1** | COSO Principle 11 | Internal and external security communication | Transactional Outbox & In-App Security Alerts | `backend/app/services/notification_service.py` |
| **CC3.1** | COSO Principle 6 | Periodic threat modeling and risk assessment | STRIDE Threat Model & DREAD scoring | `docs/PHASE_13_THREAT_MODEL.md` |
| **CC4.1** | COSO Principle 16 | Ongoing evaluation of internal controls | Automated security test suites in CI/CD | `tests/security/` |
| **CC5.1** | COSO Principle 12 | Logical access security policy enforcement | Role-Based Access Control (RBAC) & Tenant Isolation | `backend/app/core/permissions.py` |
| **CC6.1** | Logical Access | Authentication credentials and password security | Argon2id password hashing, lockout after 5 failures | `backend/app/core/security.py`, `tests/unit/test_password_security.py` |
| **CC6.2** | Logical Access | User registration and identity verification | Single-use email verification tokens (24h expiry) | `backend/app/services/auth_service.py` |
| **CC6.3** | Access Revocation | Timely de-provisioning of credentials | Token family reuse revocation, immediate logout-all | `backend/app/services/auth_service.py` |
| **CC6.6** | Edge Protection | Defenses against unauthorized edge traffic | Distributed sliding-window rate limiting (Redis) | `backend/app/core/rate_limiter.py` |
| **CC6.7** | Data in Transit | Protection of data transmitted over networks | TLS 1.3, HSTS (`max-age=63072000`), CSP, no plain HTTP | `backend/app/main.py`, `frontend/next.config.ts` |
| **CC6.8** | Malicious Software | Prevention of malicious file execution | Strict binary magic-byte validation, 10MB ceiling | `backend/app/integrations/storage/local.py` |
| **CC7.1** | Vulnerability Mgmt | Identification and remediation of vulnerabilities | Dependency scanning, SAST, automated testing | `docs/PHASE_13_VULNERABILITY_MANAGEMENT.md` |
| **CC7.2** | Security Monitoring | Detection of unauthorized security events | Centralized SHA-256 hash-chained security audit trail | `backend/app/services/security_audit_service.py` |
| **CC7.3** | Incident Response | Protocol for evaluating and responding to incidents | Formal 4-tier Incident Response Plan with runbooks | `docs/PHASE_13_INCIDENT_RESPONSE_PLAN.md` |

---

## 3. Availability Control Mapping

| Control ID | TSC Reference | Control Description | CharityChain Implementation | Evidence / Verification Pointer |
|---|---|---|---|---|
| **A1.1** | Availability | System capacity and scaling monitoring | Prometheus metrics scraping and Grafana dashboards | `docker/grafana/dashboards/app-health.json` |
| **A1.2** | Data Backup | Database and storage backup procedures | PostgreSQL pg_dump automated backup script | `docker-compose.yml` |
| **A1.3** | Disaster Recovery | Recovery time (RTO) and point (RPO) objectives | Target RTO: 4 hours; Target RPO: 1 hour | `docs/PHASE_13_SECURITY_ARCHITECTURE.md` |

---

## 4. Processing Integrity Control Mapping

| Control ID | TSC Reference | Control Description | CharityChain Implementation | Evidence / Verification Pointer |
|---|---|---|---|---|
| **PI1.1** | Completeness | Complete and accurate financial processing | Double-entry general ledger with trial balance check | `backend/app/services/ledger_service.py`, `tests/unit/test_double_entry_ledger.py` |
| **PI1.2** | Accuracy | Rounding and floating-point drift elimination | Integer minor units (`Money` VO) and Banker's rounding | `backend/app/core/currency.py`, `tests/unit/test_currency_math.py` |
| **PI1.3** | Human-in-the-Loop | Dual independent human verification | Dual-key reviewer signoff before certificate issuance | `backend/app/services/review_service.py`, `tests/security/test_dual_key_approval.py` |
| **PI1.4** | Non-Repudiation | Immutable proof of charitable achievement | Asymmetric Ed25519 digital signatures on certificates | `backend/app/services/certificate_service.py`, `tests/unit/test_certificate_signing.py` |

---

## 5. Confidentiality Control Mapping

| Control ID | TSC Reference | Control Description | CharityChain Implementation | Evidence / Verification Pointer |
|---|---|---|---|---|
| **C1.1** | Data Identification | Classification of confidential data assets | Asset classification matrix in Threat Model | `docs/PHASE_13_THREAT_MODEL.md` |
| **C1.2** | Data Destruction | Secure disposal of credentials and tokens | Cryptographic overwriting of sensitive secrets | `backend/app/services/auth_service.py` |

---

## 6. Privacy Control Mapping

| Control ID | TSC Reference | Control Description | CharityChain Implementation | Evidence / Verification Pointer |
|---|---|---|---|---|
| **P1.1** | Privacy Notice | Disclosure of data collection and usage practices | Public Transparency Methodology & Privacy Policy | `docs/PHASE_13_PRIVACY_AND_DATA_GOVERNANCE.md` |
| **P2.1** | Choice & Consent | Donor choice to remain anonymous on public records | Default donor anonymity on certificates and listings | `backend/app/services/public_verification_service.py` |
| **P3.1** | Geographic Privacy | Protection of vulnerable beneficiary locations | Centroid coarsening (~11.1 km radius) for sensitive sites | `backend/app/services/analytics/geographic_service.py` |
| **P4.1** | Telemetry Privacy | Prevention of PII leakage in observability tools | Sentry recursive PII scrubber stripping all auth/PII | `backend/app/core/sentry.py` |

