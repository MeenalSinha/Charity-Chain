# CharityChain — Phase 13 Security Hardening Guide
**Scope:** Production Deployment Hardening  
**Target Environments:** Staging, Production  
**Compliance Standard:** CIS Benchmarks, OWASP ASVS v4.0.3  

---

## 1. Container & Infrastructure Hardening

### 1.1 Non-Root Container Execution
Both `backend/Dockerfile` and `frontend/Dockerfile` must enforce non-root execution:
- **Backend**: Executes under dedicated system user `charitychain` (UID 10001, GID 10001).
- **Frontend**: Executes under user `nextjs` (UID 1001, GID 1001) using Next.js standalone output.

### 1.2 Docker Security Flags & Capabilities
In production orchestrators (Kubernetes / Docker Compose production override):
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
read_only: false
tmpfs:
  - /tmp:rw,noexec,nosuid,size=64m
```

---

## 2. FastAPI Application Hardening

### 2.1 Security Headers Middleware
All HTTP responses from FastAPI include strict headers configured in `backend/app/main.py`:
- `X-Content-Type-Options: nosniff` (prevents MIME type sniffing)
- `X-Frame-Options: DENY` (prevents clickjacking)
- `X-XSS-Protection: 0` (modern standard per OWASP to avoid audit filter vulnerabilities)
- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none';`

### 2.2 Strict CORS Configuration
CORS origins are strictly validated against configured production domains in `backend/app/core/config.py`:
- Never use wildcard `allow_origins=["*"]` when `allow_credentials=True`.
- Production enforces explicit matching (e.g., `https://charitychain.org`, `https://app.charitychain.org`).
- Disallowed origins receive `403 Forbidden` on preflight OPTIONS requests.

### 2.3 Request Payload Limits
- Maximum upload size for evidence and documents: **10 MB**.
- Maximum JSON payload size: **1 MB**.
- Binary magic byte validation inspects header bytes (`%PDF`, `\x89PNG`, `\xFF\xD8\xFF`) before reading full streams.

---

## 3. Next.js Frontend Hardening

### 3.1 Next.js Security Headers (`frontend/next.config.ts`)
Configured in `headers()` function:
```typescript
{
  key: 'X-DNS-Prefetch-Control',
  value: 'on',
},
{
  key: 'Strict-Transport-Security',
  value: 'max-age=63072000; includeSubDomains; preload',
},
{
  key: 'X-Frame-Options',
  value: 'DENY',
},
{
  key: 'X-Content-Type-Options',
  value: 'nosniff',
},
{
  key: 'Referrer-Policy',
  value: 'strict-origin-when-cross-origin',
},
{
  key: 'Permissions-Policy',
  value: 'camera=(), microphone=(), geolocation=(), payment=()',
}
```

### 3.2 BFF Anti-CSRF Guard
State-changing API calls (`POST`, `PUT`, `DELETE`, `PATCH`) in `frontend/app/api/auth/*` verify:
1. `Origin` or `Referer` matches the application's base URL.
2. Direct cross-origin browser posts are rejected.

---

## 4. Database & Cache Hardening

### 4.1 PostgreSQL Least-Privilege Architecture
- Database migrations run under a dedicated migration user (`charity_migrator`).
- Application runtime connects under `charity_app` with `SELECT`, `INSERT`, `UPDATE` grants only (no `DROP`, `TRUNCATE`, or `ALTER` permissions).
- Parameterized queries with SQLAlchemy ORM prevent SQL injection vulnerabilities.

### 4.2 Redis Cache Isolation
- Require strong authentication (`requirepass` directive).
- Protected mode enabled (`protected-mode yes`).
- Separate key namespaces:
  - `rl:*` for rate limiting (with TTLs)
  - `sess:*` for user sessions
  - `cache:*` for read acceleration

---

## 5. Secret Management & Zero-Hardcoding Enforcement

1. **No Credentials in Version Control**: `.env` is ignored in `.gitignore`; `.env.example` provides sanitized template values.
2. **High-Entropy Key Generation**:
   - `JWT_SECRET_KEY`: Minimum 32 bytes (256-bit) cryptographically random hex.
   - `WEBHOOK_SIGNING_SECRET`: Minimum 32 bytes cryptographically random hex.
   - `ED25519_PRIVATE_KEY_B64`: 32-byte Ed25519 raw seed base64 encoded.
3. **Environment Startup Validation**: Production startup validates that secrets are not default placeholders (e.g. `secret`, `change_me`, `test`).
