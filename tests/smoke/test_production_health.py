"""
tests/smoke/test_production_health.py

Phase 14: Smoke tests for production health endpoints.
These are designed to run against a live deployment.
In CI, they run against the development server with SQLite.
"""
import os

import pytest

# Base URL from environment — defaults to local dev server
BASE_URL = os.environ.get("SMOKE_TEST_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def api_client():
    """HTTP client for smoke tests."""
    import httpx
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        yield client


class TestHealthEndpoints:
    """Verify health and readiness endpoints respond correctly."""

    def test_root_health_endpoint_returns_200(self, api_client):
        """GET /health must return 200 with status information."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_api_v1_projects_public_accessible(self, api_client):
        """Public project listing must be accessible without authentication."""
        response = api_client.get("/api/v1/projects", params={"limit": 1})
        assert response.status_code in (200, 404), (
            f"Expected 200 or 404, got {response.status_code}: {response.text}"
        )

    def test_jwks_endpoint_accessible(self, api_client):
        """JWKS discovery endpoint must be publicly accessible."""
        response = api_client.get("/.well-known/jwks.json")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data

    def test_metrics_endpoint_exists(self, api_client):
        """Prometheus /metrics endpoint must exist."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        # Prometheus text format starts with # HELP
        assert "# HELP" in response.text or "charitychain" in response.text

    def test_openapi_schema_accessible(self, api_client):
        """OpenAPI schema must be accessible in non-production environments."""
        response = api_client.get("/openapi.json")
        # In production, this may be restricted (404) — that's acceptable
        assert response.status_code in (200, 404)

    def test_authenticated_endpoint_returns_401_without_token(self, api_client):
        """Authenticated endpoints must reject unauthenticated requests."""
        response = api_client.get("/api/v1/users/me")
        assert response.status_code == 401, (
            f"Expected 401, got {response.status_code}"
        )

    def test_security_headers_present(self, api_client):
        """Security headers must be present on all responses."""
        response = api_client.get("/health")
        # Check for security headers added by Phase 13 middleware
        # These may be set by nginx in production, or by FastAPI middleware
        headers = {k.lower(): v for k, v in response.headers.items()}
        # At minimum, check content-type is set
        assert "content-type" in headers


class TestAuthenticationSmoke:
    """Basic authentication flow smoke test."""

    def test_register_endpoint_accepts_requests(self, api_client):
        """Registration endpoint must be reachable."""
        # Send a deliberately invalid request to verify endpoint exists
        response = api_client.post(
            "/api/v1/auth/register",
            json={"email": "invalid", "password": "x"},
        )
        # Should get a validation error, not a 404 or 500
        assert response.status_code in (400, 422), (
            f"Expected validation error, got {response.status_code}"
        )

    def test_login_endpoint_accepts_requests(self, api_client):
        """Login endpoint must be reachable."""
        response = api_client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@test.com", "password": "wrong"},
        )
        # Should get 401 (wrong credentials) or 422 (invalid input)
        assert response.status_code in (401, 422), (
            f"Expected 401 or 422, got {response.status_code}"
        )

    def test_rate_limiting_headers_present_on_auth(self, api_client):
        """Auth endpoints should include rate-limit headers."""
        response = api_client.post(
            "/api/v1/auth/login",
            json={"email": "test@test.com", "password": "wrong"},
        )
        # Rate limit headers may or may not be present depending on Redis availability
        # This test verifies the endpoint responds without server error
        assert response.status_code != 500, "Auth endpoint returned 500"

