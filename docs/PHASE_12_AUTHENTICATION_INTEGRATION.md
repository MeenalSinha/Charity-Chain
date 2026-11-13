# CharityChain — Phase 12 Authentication Integration
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Executive Summary

CharityChain implements a secure **Backend-for-Frontend (BFF)** authentication architecture. All sensitive tokens (short-lived JWT access tokens and rotating refresh tokens) are held exclusively in server-side `HttpOnly`, `SameSite=Lax`, and `Secure` cookies. Browser JavaScript never accesses or stores authentication tokens in `localStorage`, `sessionStorage`, or client-accessible cookies, mitigating Cross-Site Scripting (XSS) token exfiltration risks.

---

## 2. Authentication Flow

```
+------------+               +-----------------+               +-----------------+
|   Browser  |               | Next.js BFF     |               | FastAPI Backend |
|  (Client)  |               | (Route Handlers)|               |  (/api/v1/auth) |
+-----+------+               +--------+--------+               +--------+--------+
      |                               |                                 |
      | 1. POST /api/auth/login       |                                 |
      |    { email, password }        |                                 |
      +------------------------------>|                                 |
      |                               | 2. POST /auth/login             |
      |                               |    { email, password }          |
      |                               +-------------------------------->|
      |                               |                                 |
      |                               | 3. Returns { access_token,      |
      |                               |    refresh_token, expires_in }  |
      |                               |<--------------------------------+
      |                               |                                 |
      | 4. Response 200 OK            |                                 |
      |    Set-Cookie: cc_access_token (HttpOnly, Secure)               |
      |    Set-Cookie: cc_refresh_token (HttpOnly, Secure)              |
      |<------------------------------+                                 |
      |                               |                                 |
      | 5. Navigates to /dashboard    |                                 |
      +------------------------------>|                                 |
      |                               | 6. Server Component reads       |
      |                               |    cookie and calls FastAPI     |
      |                               |    Authorization: Bearer <tok>  |
      |                               +-------------------------------->|
```

---

## 3. Session Security Attributes

- **Cookie Flags**:
  - `HttpOnly: true` (inaccessible to `document.cookie` / JavaScript)
  - `Secure: true` (enforced in production environments over TLS)
  - `SameSite: 'lax'` (defends against Cross-Site Request Forgery)
  - `Path: '/'`
- **Token Rotation**: Refresh tokens are single-use. When a refresh token is presented to `/api/auth/refresh`, the FastAPI backend revokes the existing family member and issues a new token pair. If token reuse is detected, the entire session family is revoked.
- **Session Revocation**: When the user signs out (`POST /api/auth/logout`), the BFF notifies the FastAPI backend to mark the session record as revoked in PostgreSQL, then deletes both cookies from the browser.

