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

    # 2. Correlation ID Middleware
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

    # 3. Global Exception Handlers
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

    return app


app = create_application()
