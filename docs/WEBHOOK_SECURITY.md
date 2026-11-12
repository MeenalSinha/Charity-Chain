# CharityChain — Webhook Security, HMAC Signatures & SSRF Protection

## 1. Overview

CharityChain treats webhook security with the highest rigor. Outbound webhook requests are cryptographically signed to prove authenticity and origin, include timestamp-based replay protection, and are strictly validated against Server-Side Request Forgery (SSRF) and DNS rebinding attacks.

---

## 2. Cryptographic HMAC-SHA256 Signatures

Every outbound HTTP request sent to a webhook endpoint includes the custom header:

```
CharityChain-Signature: t=1731450000,v1=9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
```

### Signature Computation Algorithm
1. Extract the current Unix epoch integer timestamp (`t`).
2. Construct the signed string: `{t}.{raw_request_body_bytes}`.
3. Compute the HMAC-SHA256 digest using your secret key (`whsec_...`).
4. Format the header as `t={t},v1={hex_digest}`.

### Verification Code Example (Python)

```python
import hmac
import hashlib
import time

def verify_charitychain_signature(raw_body: bytes, sig_header: str, secret: str, tolerance_seconds: int = 300) -> bool:
    try:
        parts = dict(item.split("=", 1) for item in sig_header.split(","))
        timestamp = int(parts["t"])
        signature = parts["v1"]

        # 1. Replay defense
        if abs(int(time.time()) - timestamp) > tolerance_seconds:
            return False

        # 2. Compute expected HMAC
        signed_payload = f"{timestamp}.".encode("utf-8") + raw_body
        expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()

        # 3. Constant-time comparison
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False
```

---

## 3. Server-Side Request Forgery (SSRF) Protection Engine

The SSRF protection engine (`ssrf_protection.py`) validates target URLs both at registration time and immediately before every outbound HTTP request (anti-DNS rebinding defense).

### Blocked Destinations
1. **Loopback**: `127.0.0.0/8`, `::1`, `localhost`
2. **Private RFC 1918**: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
3. **Cloud Metadata Endpoints**: `169.254.169.254` (AWS, GCP, Azure, DigitalOcean instance metadata services), `169.254.0.0/16`, `fe80::/10`
4. **Dangerous Service Ports**: SSH (22), SMTP (25), MySQL (3306), PostgreSQL (5432), Redis (6379), MongoDB (27017), Docker (2375).
5. **No Redirects**: Outbound requests strictly enforce `allow_redirects=False` to prevent open-redirect SSRF bypasses.

---

## 4. Secret Encryption at Rest

Webhook secrets are encrypted before database persistence using Fernet symmetric encryption (AES-128 in CBC mode with HMAC-SHA256 authentication) derived from platform secret keys:

```python
cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(settings.SECRET_KEY.encode()).digest()))
encrypted_secret = cipher.encrypt(raw_secret.encode()).decode()
```

Plaintext secrets are never logged or stored in unencrypted memory.

