"""
test_ssrf_protection.py - Unit Tests for SSRF Defense Engine
===========================================================
Verifies that private IP ranges, loopback addresses, cloud metadata endpoints,
and dangerous ports are strictly rejected for outbound webhook endpoints.
"""

from unittest.mock import patch
import pytest
from backend.app.core.ssrf_protection import (
    validate_webhook_url, SSRFValidationError, is_private_or_reserved_ip
)


def test_valid_public_https_url():
    """Verifies that public HTTPS endpoints with valid standard ports pass validation."""
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))]):
        validate_webhook_url("https://api.github.com/webhook")
        validate_webhook_url("https://webhook.site/abc-123")


def test_reject_insecure_scheme():
    """Verifies that plain HTTP is blocked when allow_http is False."""
    with pytest.raises(SSRFValidationError, match="HTTPS is required"):
        validate_webhook_url("http://example.com/webhook", allow_http=False)

    with pytest.raises(SSRFValidationError, match="Invalid URL scheme"):
        validate_webhook_url("ftp://example.com/webhook", allow_http=True)


def test_reject_loopback_and_localhost():
    """Verifies that localhost and 127.0.0.1 are blocked."""
    with pytest.raises(SSRFValidationError, match="loopback or private"):
        validate_webhook_url("https://127.0.0.1/webhook")

    with pytest.raises(SSRFValidationError, match="localhost or local domain"):
        validate_webhook_url("https://localhost/webhook")

    with pytest.raises(SSRFValidationError, match="localhost or local domain"):
        validate_webhook_url("https://sub.local/webhook")


def test_reject_private_rfc1918_ips():
    """Verifies that 10.x, 172.16.x, and 192.168.x are rejected."""
    for ip in ["10.0.0.1", "172.16.0.5", "192.168.1.100"]:
        with pytest.raises(SSRFValidationError, match="loopback or private"):
            validate_webhook_url(f"https://{ip}/webhook")


def test_reject_cloud_metadata_ips():
    """Verifies that AWS / GCP / Azure metadata endpoint 169.254.169.254 is rejected."""
    with pytest.raises(SSRFValidationError, match="loopback or private"):
        validate_webhook_url("https://169.254.169.254/latest/meta-data")


def test_reject_dangerous_ports():
    """Verifies that internal service ports (SSH 22, SMTP 25, Redis 6379, Postgres 5432) are blocked."""
    for port in [22, 25, 3306, 5432, 6379, 27017]:
        with pytest.raises(SSRFValidationError, match="blocked or non-standard"):
            validate_webhook_url(f"https://example.com:{port}/webhook")
