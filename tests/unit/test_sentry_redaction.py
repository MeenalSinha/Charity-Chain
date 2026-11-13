"""
test_sentry_redaction.py - Unit Tests for Sentry PII & Credential Redaction
===========================================================================
Validates recursive scrubbing of passwords, authorization headers, JWT tokens,
cookies, and user PII before telemetry egress.
"""

from backend.app.core.sentry import redact_sensitive_data, before_send_filter


def test_redact_sensitive_keys_in_dict():
    """Ensures sensitive dictionary keys are replaced with [REDACTED]."""
    payload = {
        "email": "donor@example.org",
        "password": "SuperSecretPassword123!",
        "token": "sensitive_refresh_token",
        "nested": {
            "client_secret": "paypal_secret_xyz",
            "safe_field": "public_project_id",
            "cvv": "123"
        }
    }

    sanitized = redact_sensitive_data(payload)
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["client_secret"] == "[REDACTED]"
    assert sanitized["nested"]["cvv"] == "[REDACTED]"
    assert sanitized["nested"]["safe_field"] == "public_project_id"


def test_redact_jwt_and_bearer_strings():
    """Ensures JWT patterns and Bearer strings are stripped."""
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotLeakThisSignature"
    bearer = f"Bearer {jwt}"

    assert redact_sensitive_data(jwt) == "[REDACTED_JWT]"
    assert redact_sensitive_data(bearer) == "Bearer [REDACTED]"


def test_before_send_filter_cleanses_full_event():
    """Ensures before_send callback sanitizes headers, body, user PII, and breadcrumbs."""
    event = {
        "request": {
            "headers": {
                "Authorization": "Bearer eyJhbGciOi...",
                "Cookie": "cc_access_token=xyz; cc_refresh_token=abc",
                "Host": "api.charitychain.org"
            },
            "data": {
                "password": "mypassword",
                "project_id": "proj-1"
            }
        },
        "user": {
            "id": "user-uuid-1",
            "email": "private@example.org",
            "ip_address": "192.0.2.1"
        },
        "breadcrumbs": {
            "values": [
                {"data": {"secret_token": "token-value-123"}}
            ]
        }
    }

    filtered = before_send_filter(event, hint={})

    # Headers sanitized
    assert filtered["request"]["headers"]["Authorization"] == "[REDACTED]"
    assert filtered["request"]["headers"]["Cookie"] == "[REDACTED]"
    assert filtered["request"]["headers"]["Host"] == "api.charitychain.org"

    # Body sanitized
    assert filtered["request"]["data"]["password"] == "[REDACTED]"
    assert filtered["request"]["data"]["project_id"] == "proj-1"

    # User PII pruned
    assert "email" not in filtered["user"]
    assert "ip_address" not in filtered["user"]
    assert filtered["user"]["id"] == "user-uuid-1"

    # Breadcrumbs sanitized
    assert filtered["breadcrumbs"]["values"][0]["data"]["secret_token"] == "[REDACTED]"
