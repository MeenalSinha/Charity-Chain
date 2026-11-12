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
from backend.app.api.v1.projects import router as projects_router
from backend.app.api.v1.milestones import router as milestones_router
from backend.app.api.v1.categories import router as categories_router
from backend.app.api.v1.donations import router as donations_router
from backend.app.api.v1.payments import router as payments_router
from backend.app.api.v1.refunds import router as refunds_router
from backend.app.api.v1.webhooks import router as webhooks_router
from backend.app.api.v1.funding import router as funding_router
from backend.app.api.v1.evidence import router as evidence_router
from backend.app.api.v1.verification import router as verification_router
from backend.app.api.v1.reviews import router as reviews_router
from backend.app.api.v1.payout_recipients import router as recipients_router
from backend.app.api.v1.disbursements import router as disbursements_router
from backend.app.api.v1.ledger import router as ledger_router
from backend.app.api.v1.reconciliation import router as reconciliation_router
from backend.app.api.v1.webhooks_payouts import router as webhooks_payouts_router
from backend.app.api.v1.certificates import router as certificates_router
from backend.app.api.v1.impact_ledger import router as impact_ledger_router
from backend.app.api.v1.notifications import router as notifications_router
from backend.app.api.v1.webhook_subscriptions import router as webhook_subscriptions_router

api_v1_router = APIRouter(prefix=settings.API_PREFIX)

# Mount Domain Sub-Routers
api_v1_router.include_router(auth_router)
api_v1_router.include_router(users_router)
api_v1_router.include_router(orgs_router)
api_v1_router.include_router(admin_router)
api_v1_router.include_router(projects_router)
api_v1_router.include_router(milestones_router)
api_v1_router.include_router(categories_router)
api_v1_router.include_router(donations_router)
api_v1_router.include_router(payments_router)
api_v1_router.include_router(refunds_router)
api_v1_router.include_router(webhooks_router)
api_v1_router.include_router(funding_router)
api_v1_router.include_router(evidence_router)
api_v1_router.include_router(verification_router)
api_v1_router.include_router(reviews_router)
api_v1_router.include_router(recipients_router)
api_v1_router.include_router(disbursements_router)
api_v1_router.include_router(ledger_router)
api_v1_router.include_router(reconciliation_router)
api_v1_router.include_router(webhooks_payouts_router)
api_v1_router.include_router(certificates_router)
api_v1_router.include_router(impact_ledger_router)
api_v1_router.include_router(notifications_router)
api_v1_router.include_router(webhook_subscriptions_router)


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


# Foundational contract endpoints (Phases 4-11 implement business services)
@api_v1_router.get("/donations", tags=["Donations"])
def list_donations():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Donations service contract")


@api_v1_router.get("/evidence", tags=["Evidence"])
def list_evidence():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Evidence service contract")


@api_v1_router.get("/certificates", tags=["Certificates"])
def list_certificates():
    return StandardResponse(data=[], correlation_id=correlation_id_ctx.get(), message="Certificates service contract")
