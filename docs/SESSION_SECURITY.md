# CharityChain — Session Security Specification
**Phase 2 Deliverable — Token Lifecycles, Persistent Sessions, Rotation & Replay Protection**

---

## 1. Dual-Token Architecture

CharityChain uses an authenticated dual-token session architecture:
- **JWT Access Token**: Short-lived (15 minutes), stateless bearer token containing user ID, platform role, session ID, and standard RFC 7519 claims (`iss`, `aud`, `exp`, `iat`, `jti`).
- **Persistent Refresh Token**: Long-lived (7 days), opaque, cryptographically random 48-byte token stored solely as a SHA-256 digest in the `user_sessions` database table.

---

## 2. Refresh Token Rotation & Token Family Tracking

Every time a refresh token is presented to `POST /api/v1/auth/refresh`:
1. The incoming refresh token is SHA-256 hashed and matched against `user_sessions`.
2. The current session is marked `is_revoked = True` with `revocation_reason = 'ROTATED'`.
3. A brand-new refresh token is generated and persisted in a new session sharing the identical `token_family` UUID.
4. A fresh access token is signed and returned.

### Automatic Compromise Detection (Token Reuse)
If an attacker intercepts an old, already-rotated refresh token and attempts to refresh with it:
- The backend discovers that the session was already revoked with `ROTATED`.
- **Immediate Family Revocation**: Every session sharing that `token_family` is instantly revoked (`is_revoked = True`, `revocation_reason = 'TOKEN_REUSE_DETECTED'`).
- The legitimate user and the attacker are both forced to re-authenticate.
- A critical security `AuditEvent` is recorded.

---

## 3. Session Revocation Triggers

Sessions are revoked in the following events:
1. **User Logout**: Single session revoked (`POST /api/v1/auth/logout`).
2. **Logout All**: All active sessions for that user ID are marked revoked (`POST /api/v1/auth/logout-all`).
3. **Password Reset / Change**: Every existing session for that user is automatically invalidated to prevent unauthorized persistence after credential change.
4. **Account Suspension**: When a platform administrator suspends an account (`POST /api/v1/admin/users/{id}/status`), all active sessions are revoked immediately.
5. **Token Reuse Incident**: The entire token family is revoked upon replay attempt.

---

## 4. Account Lockout & Brute-Force Defense

- **Failed Login Counter**: Every invalid password increment updates `failed_login_attempts`.
- **Lockout Threshold**: If failed attempts reach `MAX_FAILED_LOGIN_ATTEMPTS` (default: 5), `locked_until` is set to `now + 15 minutes`.
- **Fail-Fast Defense**: Subsequent attempts return `HTTP 423 Locked` without executing costly cryptographic password hashing.
- **Counter Reset**: A successful authentication resets `failed_login_attempts = 0` and clears `locked_until`.

---

## 5. Token Transport & Client-Side Guidelines

### REST API Clients & Next.js
- Access tokens should be sent via `Authorization: Bearer <token>` header.
- For browser-based SPAs, refresh tokens are stored in `HttpOnly`, `Secure`, `SameSite=Lax` cookies to prevent XSS-based token theft.
- Mutation endpoints utilizing cookies must require custom headers (e.g. `X-Requested-With` or `X-Correlation-ID`) to block Cross-Site Request Forgery (CSRF).

### Streamlit Transitional Interface
- Tokens are stored exclusively in the server-side Python runtime memory (`st.session_state`).
- Tokens are never exposed in browser URLs, query parameters, or client-rendered HTML DOM nodes.

