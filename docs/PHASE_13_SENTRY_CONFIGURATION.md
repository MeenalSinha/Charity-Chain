# CharityChain — Phase 13 Sentry Error Monitoring Configuration
**Document Version:** 1.0.0  
**Implementation Files:**  
- Backend: `backend/app/core/sentry.py`  
- Frontend Client: `frontend/sentry.client.config.ts`  
- Frontend Server: `frontend/sentry.server.config.ts`  

---

## 1. Zero-PII & Zero-Credential Policy

Under SOC 2, GDPR, and CharityChain's Privacy Policy, **no Personally Identifiable Information (PII) or authentication credentials may ever be transmitted to external monitoring platforms**.

The Sentry integration implements automated recursive sanitization across all request envelopes, exceptions, and breadcrumbs.

---

## 2. Redacted Fields & Patterns

The following keys and headers are recursively stripped or replaced with `[REDACTED]` before event transmission:

| Category | Keys / Headers / Patterns | Redaction Treatment |
|---|---|---|
| **Authentication Headers** | `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key` | Replaced with `[REDACTED]` |
| **Passwords & Secrets** | `password`, `secret`, `client_secret`, `signing_key`, `private_key` | Replaced with `[REDACTED]` |
| **Tokens & Sessions** | `token`, `access_token`, `refresh_token`, JWT string patterns | Replaced with `[REDACTED_JWT]` |
| **Payment & Financial** | `cvv`, `credit_card`, `card_number` | Replaced with `[REDACTED]` |
| **User Identifiers** | `email`, `username`, `ip_address` | Completely pruned from user context |

---

## 3. Implementation Details

### 3.1 Backend `before_send` Hook
In `backend/app/core/sentry.py`:
```python
def before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Recursively sanitizes headers, POST data, breadcrumbs, and user objects
    ...
```

### 3.2 Performance Tracing & Profiling
Configured in `backend/app/core/config.py`:
- `SENTRY_TRACES_SAMPLE_RATE`: Default `0.1` (10% sampling) to minimize overhead.
- `SENTRY_PROFILES_SAMPLE_RATE`: Default `0.1` (10% profiling on sampled transactions).
