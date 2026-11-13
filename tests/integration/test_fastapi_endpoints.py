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
    list_donations,
    list_evidence,
    list_certificates,
)
from backend.app.api.v1.projects import discover_projects as list_projects
from backend.app.api.v1.milestones import list_all_milestones as list_milestones
from backend.app.api.schemas import StandardResponse, HealthResponse


def test_api_health_endpoint():
    """Verify health check returns valid HealthResponse envelope."""
    response = get_health()
    assert response.success is True
    assert isinstance(response.data, HealthResponse)
    assert response.data.version == "2.0.0"
    assert response.data.status.upper() == "HEALTHY"
    assert response.data.environment in ("development", "testing", "staging", "production")


def test_api_db_health_endpoint():
    """Verify database connection health check."""
    response = get_db_health()
    assert response.success is True
    assert isinstance(response.data, dict)
    assert response.data.get("database_connected") is True


def test_api_foundational_endpoints(db_session):
    """Verify foundational domain endpoints return standard enveloped lists."""
    proj_resp = list_projects(db=db_session)
    assert proj_resp.success is True
    assert isinstance(proj_resp.data, list)

    mile_resp = list_milestones(db=db_session)
    assert mile_resp.success is True
    assert isinstance(mile_resp.data, list)

    for fn in [list_donations, list_evidence, list_certificates]:
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
