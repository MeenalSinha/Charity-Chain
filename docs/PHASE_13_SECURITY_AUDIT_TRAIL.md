# CharityChain — Phase 13 Security Audit Trail Architecture
**Document Version:** 1.0.0  
**Implementation Files:**  
- Model: `backend/app/db/models/security_audit.py`  
- Service: `backend/app/services/security_audit_service.py`  
- Migration: `backend/alembic/versions/012_phase13_security_audit_and_observability.py`  

---

## 1. Architectural Distinction & Independence

A core architectural principle of CharityChain is the strict physical and logical separation between the two cryptographic ledgers:

1. **Proof-of-Impact Ledger (`impact_ledger_*`)**:
   - Purpose: Verifiable evidence and milestone completion attestations for donors and the public.
   - Cryptography: Ed25519 digital signatures and signed checkpoint receipts.
2. **Security Audit Trail (`security_audit_events`)**:
   - Purpose: Non-repudiation, administrative accountability, identity tracking, intrusion detection, and SOC 2 / ISO 27001 compliance.
   - Cryptography: SHA-256 sequential hash chaining.

---

## 2. Cryptographic Hash-Chaining Formula

Each `SecurityAuditEvent` incorporates the hash of its immediate predecessor:

$$\text{event\_hash}_n = \text{SHA-256}(\text{previous\_hash}_{n-1} \parallel \text{event\_id}_n \parallel \text{event\_type}_n \parallel \text{action}_n \parallel \text{created\_at\_iso}_n \parallel \text{RFC8785}(\text{details}_n))$$

Where:
- $\text{previous\_hash}_0 = \text{"0" * 64}$ (Genesis Block)
- $\text{RFC8785}(\text{details})$: Deterministic canonical JSON serialization with sorted keys and UTF-8 encoding.

### 2.1 Tamper-Evidence Guarantees
- **Record Alteration**: Modifying any field (such as `action` or `details`) alters the calculated hash and fails the verification check.
- **Record Deletion**: Deleting record $k$ breaks the link between record $k-1$ and record $k+1$.
- **Record Insertion**: Injecting a record between $k$ and $k+1$ invalidates the `previous_hash` expected by $k+1$.

---

## 3. Canonical Event Types Catalog

| Event Type | Typical Severity | Trigger Condition |
|---|:---:|---|
| `AUTH_LOGIN_SUCCESS` | `INFO` | Successful user authentication via password or session token |
| `AUTH_LOGIN_FAILURE` | `WARNING` | Invalid credentials provided |
| `AUTH_LOCKOUT` | `ERROR` | Account locked out after 5 consecutive failed attempts |
| `AUTH_TOKEN_ROTATION` | `INFO` | Valid single-use refresh token exchanged for new token pair |
| `AUTH_TOKEN_REUSE_BREACH` | `CRITICAL` | Attempt to use an already-consumed refresh token (session family revoked) |
| `AUTH_PASSWORD_RESET_REQUEST` | `INFO` | Password reset initiated |
| `AUTH_PASSWORD_CHANGED` | `WARNING` | User password successfully modified |
| `USER_ROLE_CHANGED` | `CRITICAL` | User platform role escalated or downgraded |
| `ORGANIZATION_ROLE_CHANGED` | `WARNING` | Membership role modified (e.g. Member to Admin) |
| `API_RATE_LIMIT_EXCEEDED` | `WARNING` | Client exceeded sliding-window rate limit |
| `EVIDENCE_TAMPERING_DETECTED` | `ERROR` | ELA mean deviation or pHash duplicate anomaly detected |
| `SECURITY_POLICY_UPDATED` | `CRITICAL` | System security settings modified |
| `ADMIN_ACTION` | `WARNING` | Administrative user modification, freeze, or override |

---

## 4. Verification and Admin API Endpoints

The security audit service exposes verification and inspection capabilities guarded by `require_platform_role(PlatformRole.PLATFORM_ADMIN)`:

- `GET /api/v1/admin/security/audit-logs`: Paginated listing with filtering by `event_type`, `severity`, `user_id`, and date range.
- `GET /api/v1/admin/security/audit-logs/verify`: Executes sequential hash verification across the audit chain and returns real-time validity status.

