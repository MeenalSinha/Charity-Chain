"""
test_fastapi_endpoints.py - Integration tests for FastAPI endpoints & router
=============================================================================
Tests foundational endpoints, route registry, schemas, and response envelopes
without thread/event-loop deadlocks on Windows/Python 3.14.
"""

import pytest
from backend.app.main import app
from backend.app.api.v1.router import (
    get_health,
    get_db_health,
    list_projects,
    list_milestones,
    list_donations,
    list_evidence,
    list_certificates,
)
from backend.app.api.schemas import StandardResponse, HealthResponse


def test_api_health_endpoint():
    """Verify health check returns valid HealthResponse envelope."""
    response = get_health()
    assert response.success is True
    assert isinstance(response.data, HealthResponse)
    assert response.data.version == "2.0.0"
    assert response.data.status.upper() == "HEALTHY"
    assert response.data.environment == "development"


def test_api_db_health_endpoint():
    """Verify database connection health check."""
    response = get_db_health()
    assert response.success is True
    assert isinstance(response.data, dict)
    assert response.data.get("database_connected") is True


def test_api_foundational_endpoints():
    """Verify foundational domain endpoints return standard enveloped lists."""
    endpoints = [
        list_projects,
        list_milestones,
        list_donations,
        list_evidence,
        list_certificates,
    ]
    for fn in endpoints:
        resp = fn()
        assert resp.success is True
        assert isinstance(resp.data, list)


def test_app_routes_registered():
    """Verify FastAPI application has all required v1 routes registered in OpenAPI."""
    openapi_paths = list(app.openapi()["paths"].keys())
    expected_paths = [
        "/api/v1/health",
        "/api/v1/health/db",
        "/api/v1/projects",
        "/api/v1/milestones",
        "/api/v1/donations",
        "/api/v1/evidence",
        "/api/v1/certificates",
    ]
    for expected in expected_paths:
        assert expected in openapi_paths, f"Missing registered route: {expected}"


def test_app_middleware_and_openapi():
    """Verify app middleware configuration and OpenAPI schema generation."""
    openapi_schema = app.openapi()
    assert openapi_schema["info"]["title"] == "CharityChain"
    assert openapi_schema["info"]["version"] == "2.0.0"
    assert "/api/v1/health" in openapi_schema["paths"]
    assert "/api/v1/health/db" in openapi_schema["paths"]
    assert "/api/v1/projects" in openapi_schema["paths"]
