# CharityChain — Authentication & Identity Specification
**Phase 2 Deliverable — Authentication Lifecycles, Argon2id, JWTs & Persistent Sessions**

---

## 1. Authentication Architecture Overview

CharityChain's authentication system establishes verified identity across donors, NGO staff, independent reviewers, and platform administrators. Built on zero-trust principles:
- **No Autonomous Elevation**: Public registration automatically assigns the default unprivileged `DONOR` role.
- **Argon2id Password Security**: Passwords are never stored in plaintext or reversible formats. Salting and hashing utilize RFC 9106 Argon2id.
- **Dual-Token Architecture**: Short-lived JWT access tokens (15 minutes) paired with persistent server-tracked refresh tokens (7 days).
- **Compromise Detection**: Refresh token rotation tracks token families and automatically revokes all associated sessions if a rotated token is reused.
- **Anti-Enumeration Protections**: Login, password reset, and verification resend endpoints return generic status messages to prevent user enumeration.

---

## 2. User Registration Workflow

```
USER
 │
 ├── 1. POST /api/v1/auth/register (full_name, email, password, password_confirm)
 ▼
FastAPI Backend
 │
 ├── 2. Validate email format & password length (8-128 chars)
 ├── 3. Normalize email (lowercase, stripped)
 ├── 4. Check case-insensitive uniqueness in DB
 ├── 5. Hash password using Argon2id
 ├── 6. Create User (role=DONOR, status=PENDING_VERIFICATION, is_verified=False)
 ├── 7. Create UserProfile
 ├── 8. Generate 32-byte secure token, store SHA-256 hash in email_verification_tokens
 ├── 9. Dispatch email via Notification Provider (Dev sink / SMTP)
 ├── 10. Record AuditEvent (USER_REGISTERED)
 ▼
Returns StandardResponse[RegisterResponse] (requires_email_verification=True)
```

### Registration Constraints
- **Role Parameter Ignored**: Clients cannot specify a `role` during registration. Privileged roles (`REVIEWER`, `PLATFORM_ADMIN`) can only be provisioned via administrative channels or the secure CLI.
- **Case-Insensitive Uniqueness**: `user@charity.org` and `USER@charity.org` map to the identical account.

---

## 3. Email Verification Lifecycle

```
USER
 │
 ├── 1. Opens link containing single-use token
 ├── 2. POST /api/v1/auth/verify-email (token)
 ▼
FastAPI Backend
 │
 ├── 3. SHA-256 hash token & lookup unused, non-expired token record
 ├── 4. Mark token is_used=True, used_at=now
 ├── 5. Update User: is_verified=True, account_status=ACTIVE, email_verified_at=now
 ├── 6. Record AuditEvent (USER_EMAIL_VERIFIED)
 ▼
Returns StandardResponse (verified=True)
```

Unverified accounts cannot authenticate or access protected endpoints (`HTTP 403 Forbidden: Email address is not verified`).

---

## 4. Secure Login & Brute-Force Protection

```
USER
 │
 ├── 1. POST /api/v1/auth/login (email, password)
 ▼
FastAPI Backend
 │
 ├── 2. Normalize email & lookup User
 ├── 3. Check account lockout:
 │       └── If locked_until > now: Return HTTP 423 (Locked)
 ├── 4. Check status:
 │       └── If SUSPENDED / DEACTIVATED: Return HTTP 403 (Disabled)
 ├── 5. Verify password hash using constant-time Argon2id:
 │       ├── Match FAILED: increment failed_login_attempts.
 │       │   └── If attempts >= 5: lock account for 15 minutes. Return HTTP 401.
 │       └── Match PASSED: reset failed_login_attempts=0, locked_until=None.
 ├── 6. Check is_verified:
 │       └── If False: Return HTTP 403 (Email not verified)
 ├── 7. Generate random 48-byte refresh token, hash with SHA-256
 ├── 8. Persist UserSession with new token_family UUID
 ├── 9. Sign JWT access token (15-min expiry) with user_id, role, session_id
 ├── 10. Record AuditEvent (USER_LOGIN_SUCCESS)
 ▼
Returns StandardResponse[TokenResponse] (access_token, refresh_token, user_details)
```

---

## 5. Refresh Token Rotation & Compromise Detection

To prevent replay attacks and stolen session exploitation, refresh tokens are strictly single-use.

```
CLIENT
 │
 ├── 1. POST /api/v1/auth/refresh (refresh_token)
 ▼
FastAPI Backend
 │
 ├── 2. Compute SHA-256 hash of refresh_token
 ├── 3. Lookup UserSession by refresh_token_hash
 │
 ├── 4. CASE A: Session is already revoked / rotated!
 │       ├── SECURITY ALERT: Refresh token reuse detected!
 │       ├── Revoke ALL sessions belonging to the same token_family
 │       ├── Record high-severity AuditEvent (TOKEN_REUSE_DETECTED)
 │       └── Return HTTP 401 Unauthorized (Token compromised)
 │
 ├── 5. CASE B: Session is expired (expires_at < now)
 │       └── Mark session revoked (reason=EXPIRED), Return HTTP 401
 │
 └── 6. CASE C: Session is valid
         ├── Mark current session is_revoked=True, reason="ROTATED"
         ├── Issue fresh 48-byte refresh token in the SAME token_family
         ├── Create new UserSession
         ├── Sign fresh 15-minute access token
         └── Return StandardResponse[TokenResponse]
```

---

## 6. Password Recovery & Reset

1. **Request Reset (`POST /api/v1/auth/forgot-password`)**:
   - User submits email.
   - If user exists and is active, invalidates previous unused reset tokens, generates a 32-byte secure token, stores its SHA-256 digest with 30-minute expiration, and dispatches reset instructions.
   - Endpoint returns a generic success message regardless of email existence to prevent user enumeration.
2. **Execute Reset (`POST /api/v1/auth/reset-password`)**:
   - User submits recovery token and new password.
   - Validates token hash and expiration.
   - Validates password length and complexity.
   - Hashes new password with Argon2id and updates `password_changed_at`.
   - **Session Invalidation**: Automatically revokes **all active sessions** for this user.
   - Records audit event `USER_PASSWORD_RESET`.

