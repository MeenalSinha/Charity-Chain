"""
sentry.py - Production Error Monitoring & Distributed Tracing
=============================================================
Initializes Sentry SDK with strict PII and credential redaction.
Ensures zero sensitive data (passwords, tokens, cookies, auth headers)
is transmitted to external observability platforms.
"""

import re
import logging
from typing import Dict, Any, Optional
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from backend.app.core.config import settings

logger = logging.getLogger("charitychain.sentry")

# Sensitive key names to redact recursively
SENSITIVE_KEYS = {
    "password", "secret", "token", "access_token", "refresh_token",
    "authorization", "cookie", "set-cookie", "client_secret",
    "private_key", "signing_key", "cvv", "credit_card", "card_number"
}


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redacts sensitive dictionary keys, list elements, or strings."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            k_lower = str(key).lower()
            if any(s in k_lower for s in SENSITIVE_KEYS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = redact_sensitive_data(value)
        return sanitized
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    elif isinstance(data, str):
        # Redact JWT tokens
        if re.match(r"^ey[A-Za-z0-9_-]+\.ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", data):
            return "[REDACTED_JWT]"
        # Redact Bearer headers
        if data.lower().startswith("bearer "):
            return "Bearer [REDACTED]"
        return data
    return data


def before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sentry before_send callback.
    Enforces the zero-PII/zero-credential privacy policy by sanitizing the entire event payload.
    """
    try:
        # 1. Sanitize Request Headers
        if "request" in event and "headers" in event["request"]:
            headers = event["request"]["headers"]
            for h in list(headers.keys()):
                if h.lower() in ("authorization", "cookie", "set-cookie", "x-api-key"):
                    headers[h] = "[REDACTED]"

        # 2. Sanitize Request Data / Body
        if "request" in event and "data" in event["request"]:
            event["request"]["data"] = redact_sensitive_data(event["request"]["data"])

        # 3. Sanitize User Object (Never expose emails or real names to Sentry)
        if "user" in event:
            user = event["user"]
            if isinstance(user, dict):
                user.pop("email", None)
                user.pop("username", None)
                user.pop("ip_address", None)

        # 4. Sanitize Breadcrumbs
        if "breadcrumbs" in event and "values" in event["breadcrumbs"]:
            for breadcrumb in event["breadcrumbs"]["values"]:
                if "data" in breadcrumb:
                    breadcrumb["data"] = redact_sensitive_data(breadcrumb["data"])

        # 5. Sanitize Extra context
        if "extra" in event:
            event["extra"] = redact_sensitive_data(event["extra"])

        return event

    except Exception as exc:
        logger.warning(f"Error during Sentry event redaction: {exc}")
        return event


def init_sentry():
    """Initializes the Sentry SDK if SENTRY_DSN is configured."""
    if not settings.SENTRY_DSN:
        logger.info("Sentry DSN not configured; error monitoring disabled.")
        return

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
        release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
        before_send=before_send_filter,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
        ],
        send_default_pii=False,
    )
    logger.info("Sentry error monitoring initialized with strict PII redaction.")
