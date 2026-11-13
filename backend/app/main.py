"""
main.py - Modular FastAPI Application Entry Point
=================================================
Initializes FastAPI, configures CORS, correlation IDs, structured logging,
and global exception handling.
"""

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from backend.app.core.config import settings
from backend.app.core.logging import logger, correlation_id_ctx, user_id_ctx
from backend.app.core.exceptions import CharityChainException
from backend.app.api.schemas import StandardErrorResponse, ErrorDetail
from backend.app.api.v1.router import api_v1_router
from backend.app.db.session import check_db_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: startup and graceful shutdown."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    try:
        from backend.app.core.sentry import init_sentry
        init_sentry()
    except Exception as exc:
        logger.warning(f"Sentry initialization skipped or failed: {exc}")

    db_ok = check_db_connection()
    if db_ok:
        logger.info("Database connectivity established successfully.")
    else:
        logger.warning("Database connectivity check failed during startup.")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_application() -> FastAPI:
    """Factory function creating and configuring the FastAPI app."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="CharityChain — Evidence-Backed Charitable Giving & Impact Verification API",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Security Response Headers Middleware (OWASP ASVS Hardening)
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        return response

    # 3. Prometheus Operational Telemetry Middleware
    @app.middleware("http")
    async def prometheus_telemetry_middleware(request: Request, call_next):
        if not settings.METRICS_ENABLED or request.url.path == "/metrics":
            return await call_next(request)
        import time
        from backend.app.core.metrics import http_requests_total, http_request_duration_seconds
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
        return response

    # 4. Correlation ID Middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        token = correlation_id_ctx.set(correlation_id)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            correlation_id_ctx.reset(token)

    # 5. Global Exception Handlers
    from backend.app.core.rate_limiter import RateLimitExceededException

    @app.exception_handler(RateLimitExceededException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceededException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers
        )
    @app.exception_handler(CharityChainException)
    async def custom_exception_handler(request: Request, exc: CharityChainException):
        logger.warning(
            f"Handled application exception: {exc.code} - {exc.message}",
            extra={"extra_data": exc.details}
        )
        error_payload = StandardErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
            correlation_id=correlation_id_ctx.get()
        )
        return JSONResponse(status_code=exc.status_code, content=error_payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.info(f"Request validation failure on {request.url.path}: {exc.errors()}")
        error_payload = StandardErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request body or query parameters failed validation.",
                details={"errors": exc.errors()}
            ),
            correlation_id=correlation_id_ctx.get()
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_payload.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
        error_payload = StandardErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred."
            ),
            correlation_id=correlation_id_ctx.get()
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_payload.model_dump(mode="json"))

    # 4. Mount Master Router
    app.include_router(api_v1_router)

    # 5. Root JWKS Discovery Endpoint (RFC 7517)
    from backend.app.db.session import get_db
    from backend.app.services.key_management_service import KeyManagementService
    from fastapi import Depends
    from sqlalchemy.orm import Session

    @app.get("/.well-known/jwks.json", tags=["JWKS Discovery"])
    def root_jwks(db: Session = Depends(get_db)):
        return KeyManagementService.build_jwks(db)

    # 6. Prometheus Metrics Endpoint
    from fastapi.responses import Response
    from backend.app.core.metrics import get_latest_metrics, CONTENT_TYPE_LATEST

    @app.get("/metrics", tags=["Observability"])
    def prometheus_metrics():
        """Exposes standard Prometheus metrics in text format."""
        return Response(content=get_latest_metrics(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_application()
