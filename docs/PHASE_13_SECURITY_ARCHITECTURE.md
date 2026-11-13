# CharityChain — Phase 13 Security Architecture
**Document Version:** 1.0.0  
**Status:** Approved for Production Deployment  
**Audience:** Platform Architects, Security Engineers, DevOps Engineers  

---

## 1. Architectural Philosophy: Defense-in-Depth & Zero-Trust

CharityChain's security architecture operates on the principle of **Defense-in-Depth** and **Zero-Trust**. Every network request, internal component interaction, and persistent write is verified, authenticated, and authorized regardless of where it originates.

```
                    [ 1. Network & Edge Layer ]
           * HTTPS (TLS 1.3), HSTS, Security Headers, CSP
           * Next.js BFF Reverse Proxy & Session Cookies
                               |
                               v
                     [ 2. API Gateway Layer ]
           * Redis-backed Distributed Sliding-Window Rate Limiter
           * Anti-Enumeration & Account Lockout Defense
           * Request Size Ceilings & Magic-Byte Validation
                               |
                               v
                 [ 3. Identity & Access Layer ]
           * Argon2id Cryptographic Password Storage
           * JWT with Single-Use Refresh Token Rotation
           * Fine-Grained Role-Based Access Control (RBAC)
           * Tenant Isolation & Conflict-of-Interest Guards
                               |
                               v
                   [ 4. Application Logic Layer ]
           * Currency-Safe Financial Contracts (Integer Minor Units)
           * Dual-Key Human Approval for Milestone Completion
           * Zero-Trust SSRF Validation on Outbound Webhooks
           * Sentry Telemetry with Recursive PII Redaction
                               |
                               v
                   [ 5. Data & Storage Layer ]
           * SHA-256 Hash-Chained Security Audit Trail
           * Independent Cryptographic Proof-of-Impact Ledger
           * Double-Entry General Ledger Balance Invariant
           * Encrypted Webhook Secrets (AES-256 Fernet)
```

---

## 2. Layer 1: Network & Edge Security

### 2.1 HTTP Security Response Headers
Every response served by the Next.js frontend and FastAPI backend must include mandatory security headers:
- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' http://localhost:8000; frame-ancestors 'none';`

### 2.2 BFF Session Security
- Cookies use `HttpOnly`, `SameSite=Lax`, and `Secure` (in production).
- Browser client code never interacts directly with raw backend JWT tokens.
- Token refresh happens server-side within the Next.js route handler (`/api/auth/refresh`).

---

## 3. Layer 2: Rate Limiting & Abuse Prevention

### 3.1 Distributed Sliding Window Architecture
Rate limiting is enforced using Redis sorted sets (`ZSET`) to achieve a high-precision sliding window:
- Each incoming request records its Unix millisecond timestamp as both score and member.
- Expired timestamps older than `(now - window_seconds)` are trimmed atomically via `ZREMRANGEBYSCORE`.
- Current count is evaluated via `ZCARD`.
- If `count >= limit`, the request is rejected with `HTTP 429 Too Many Requests`.

### 3.2 Granular Tier Limits

| Tier / Endpoint Pattern | Default Limit | Window | Purpose |
|---|:---:|:---:|---|
| **Auth Strict** (`/api/v1/auth/login`, `/register`, `/forgot-password`) | 5 req | 60 sec | Brute-force and credential stuffing prevention |
| **Financial** (`/api/v1/donations/*`) | 20 req | 60 sec | Payment intent and carding fraud prevention |
| **Evidence & AI** (`/api/v1/evidence/*`) | 10 req | 60 sec | Resource exhaustion and compute abuse protection |
| **Public Transparency** (`/api/v1/transparency/*`, `/projects`) | 60 req | 60 sec | Scraper throttling and availability guarantee |
| **Global Default** (All other endpoints) | 100 req | 60 sec | Platform stability baseline |

### 3.3 Fail-Safe Degradation
If Redis is temporarily unreachable, the rate limiter falls back to an in-memory thread-safe dictionary cache with log warning rather than halting critical operations.

---

## 4. Layer 3: Identity, Credentials & Authorization

### 4.1 Argon2id Password Cryptography
- Memory Cost: 65,536 KiB (64 MB)
- Time Cost: 3 iterations
- Parallelism: 4 threads
- Constant-time string comparison preventing side-channel timing attacks.

### 4.2 Refresh Token Rotation & Reuse Detection
- Refresh tokens are strictly single-use.
- When an expired or already-consumed token is presented, the system triggers a **Security Breach Event** and immediately revokes all sessions in the token family.

### 4.3 Tenant Isolation & RBAC
- Organization resources require verified membership matching `organization_id`.
- Reviewers cannot review projects from organizations with which they have active or historical affiliations (Anti-Conflict of Interest).

---

## 5. Layer 4: Cryptographic Key Management & Telemetry

### 5.1 Asymmetric Ed25519 Certificate Signing Keys
- Keys generated using `cryptography.hazmat.primitives.asymmetric.ed25519`.
- Private keys stored in environment variables (`ED25519_PRIVATE_KEY_B64`) or secure KMS.
- Public key published at `GET /.well-known/jwks.json` for universal third-party verification.

### 5.2 Sentry Error Redaction (Zero PII Policy)
- Sentry SDK integrated with a custom `before_send` hook.
- Recursively removes authorization headers, cookies, passwords, refresh tokens, credit card fields, and emails before sending events to external monitoring.

---

## 6. Layer 5: Tamper-Evident Security Audit Trail

### 6.1 Strict Architectural Separation
CharityChain maintains two entirely distinct cryptographic ledgers:
1. **Proof-of-Impact Ledger (`impact_ledger_*`)**: Records verified milestone completions, donor contributions, and impact achievements.
2. **Security Audit Trail (`security_audit_*`)**: Records authentication events, privilege escalations, rate-limit violations, administrative actions, and configuration changes.

### 6.2 SHA-256 Hash Chaining
Every `SecurityAuditEvent` record computes:
$$\text{event\_hash} = \text{SHA-256}(\text{previous\_hash} + \text{event\_type} + \text{user\_id} + \text{timestamp} + \text{payload\_json})$$

The genesis record uses `GENESIS_SECURITY_AUDIT_BLOCK`. Any retroactive modification, deletion, or insertion breaks the chain and is flagged immediately by automated integrity audits.

