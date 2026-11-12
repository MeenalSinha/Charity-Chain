# CharityChain — Email Provider Configuration & Transport Setup

## 1. Overview

CharityChain features an extensible, enterprise-ready email delivery architecture providing reliable transactional email delivery with TLS encryption, MIME multipart encoding, and structured audit telemetry.

---

## 2. Supported Email Providers

The provider is configured via `EMAIL_PROVIDER`:

1. **`smtp` (Production Standard)**:
   - Connects to standard SMTP gateways (SendGrid, Amazon SES, Mailgun, Postmark, or dedicated MTA).
   - Supports opportunistic STARTTLS (port 587) or implicit SSL/TLS (port 465).
   - Enforces configurable connection and socket timeouts (`SMTP_TIMEOUT_SECONDS`).
   - Automatically falls back to plain-text alternative if HTML rendering is unsupported by recipient client.

2. **`dev_sink` (Development & Testing Default)**:
   - Captures sent emails into an in-memory inspectable log (`DevNotificationSink.outbox`).
   - Safe for offline unit and integration tests.
   - **Fail-Closed Protection**: In `production` environments, attempts to send through `dev_sink` without configured credentials abort with `NotificationDeliveryException` to prevent silent delivery blackholes.

---

## 3. Environment Variables & Settings

Add the following configuration keys to your `.env` or deployment secrets:

```env
# Email Provider Selection ('smtp' or 'dev_sink')
EMAIL_PROVIDER=smtp

# SMTP Gateway Credentials
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USERNAME=apikey
SMTP_PASSWORD=your_sendgrid_or_ses_smtp_key_here
SMTP_USE_TLS=true
SMTP_USE_SSL=false
SMTP_TIMEOUT_SECONDS=10.0

# Sender Identity
EMAIL_FROM_ADDRESS=notifications@charitychain.org
EMAIL_FROM_NAME="CharityChain Verified Impact"
```

---

## 4. Health Check & Diagnostics

The email provider exposes a diagnostic connection verification method:

```python
from backend.app.integrations.notifications.factory import get_email_provider

provider = get_email_provider()
is_healthy, status_message = provider.verify_connection()
if not is_healthy:
    logger.error(f"Email subsystem connection failed: {status_message}")
```

---

## 5. Security & Deliverability Best Practices

1. **MIME Multipart Encoding**: All transactional messages are compiled with `multipart/alternative`, including both `text/html` and a pure UTF-8 `text/plain` fallback to optimize anti-spam scores.
2. **Strict HTML Escaping**: All variables injected into email templates are sanitized through Python's standard `html.escape` to block script and CSS injection attacks.
3. **No Credential Logging**: SMTP passwords and API tokens are suppressed from log outputs.
4. **Audit Trail**: Every outbound dispatch attempt is recorded in `notification_delivery_attempts` with response codes, delivery status, and timestamps.

