"""
router.py - Version 1 API Master Router
=======================================
Mounts all domain sub-routers and foundational health check endpoints under /api/v1.
"""

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.core.logging import correlation_id_ctx
from backend.app.api.schemas import StandardResponse, HealthResponse
from backend.app.db.session import check_db_connection

from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.users import router as users_router
from backend.app.api.v1.organizations import router as orgs_router
from backend.app.api.v1.admin import router as admin_router

api_v1_router = APIRouter(prefix=settings.API_PREFIX)

# Mount Phase 2 Authentication, Identity & Access Control sub-routers
api_v1_router.include_router(auth_router)
api_v1_router.include_router(users_router)
api_v1_router.include_router(orgs_router)
api_v1_router.include_router(admin_router)


@api_v1_router.get(
    "/health",
    response_model=StandardResponse[HealthResponse],
    summary="Application Health Status",
    tags=["Health & System"]
)
def get_health():
    """Returns application health, version, and environment information."""
    db_ok = check_db_connection()
    health_data = HealthResponse(
        status="HEALTHY" if db_ok else "DEGRADED",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_ok
    )
    return StandardResponse(
        data=health_data,
        correlation_id=correlation_id_ctx.get(),
        message="CharityChain service is operational."
    )


@api_v1_router.get(
    "/health/db",
    response_model=StandardResponse[dict],
    summary="Database Connectivity Check",
    tags=["Health & System"]
)
def get_db_health():
    """Dedicated database connection health check."""
    connected = check_db_connection()
    return StandardResponse(
        data={"database_connected": connected},
        correlation_id=correlation_id_ctx.get(),
        message="Database connection verified." if connected else "Database connection unavailable."
    )


# Foundational contract endpoints (Phases 2-11 implement business services)
@api_v1_router.get("/projects", tags=["Projects"])
def list_projects():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Projects service contract")


@api_v1_router.get("/milestones", tags=["Milestones"])
def list_milestones():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Milestones service contract")


@api_v1_router.get("/donations", tags=["Donations"])
def list_donations():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Donations service contract")


@api_v1_router.get("/evidence", tags=["Evidence"])
def list_evidence():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Evidence service contract")


@api_v1_router.get("/certificates", tags=["Certificates"])
def list_certificates():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Certificates service contract")
