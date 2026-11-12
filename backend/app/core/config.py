"""
config.py - Centralized, Typed & Validated Configuration Management
===================================================================
Uses Pydantic Settings for type safety and environment variable binding.
Adheres strictly to the Zero-Hardcoding Policy:
Every environment-dependent and operational threshold is configurable.
"""

from typing import List, Optional, Literal
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # 1. Environment & Server
    ENVIRONMENT: Literal["development", "testing", "staging", "production"] = Field(
        default="development",
        description="Current runtime environment"
    )
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    APP_NAME: str = Field(default="CharityChain", description="Application Name")
    APP_VERSION: str = Field(default="2.0.0", description="Application Version")
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    
    API_HOST: str = Field(default="127.0.0.1", description="API listen host")
    API_PORT: int = Field(default=8000, description="API listen port")
    FRONTEND_PORT: int = Field(default=8501, description="Streamlit / Web frontend port")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"],
        description="Allowed CORS origin URLs"
    )

    # 2. Database & Persistence (SQLAlchemy PostgreSQL / SQLite async fallback for tests)
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/charity_app.db",
        description="SQLAlchemy async connection string"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Max overflow connections")
    DATABASE_POOL_TIMEOUT_SECONDS: int = Field(default=30, description="Connection timeout")
    TINYDB_PATH: Path = Field(default=Path("data/charity.db"), description="Legacy TinyDB path")

    # 3. Security & Authentication
    SECRET_KEY: str = Field(
        default="development_insecure_secret_key_change_in_production_1234567890",
        description="JWT cryptographic signing secret"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, description="Token validity window (default 15 mins)")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token validity")
    MAX_FAILED_LOGIN_ATTEMPTS: int = Field(default=5, description="Attempts before temporary lockout")
    ACCOUNT_LOCKOUT_MINUTES: int = Field(default=15, description="Duration of account lockout in minutes")
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = Field(default=24, description="Email verification token validity")
    PASSWORD_RESET_EXPIRE_MINUTES: int = Field(default=30, description="Password reset token validity")
    INVITATION_EXPIRE_DAYS: int = Field(default=7, description="Organization invitation validity")
    PASSWORD_MIN_LENGTH: int = Field(default=8, description="Minimum password length")
    PASSWORD_MAX_LENGTH: int = Field(default=128, description="Maximum password length")

    # 4. Storage & Files
    STORAGE_PROVIDER: Literal["local", "s3", "minio", "pinata"] = Field(
        default="local",
        description="Selected file storage backend"
    )
    STORAGE_LOCAL_DIR: Path = Field(
        default=Path("data/evidence"),
        description="Local directory for evidence storage"
    )
    STORAGE_DOCUMENTS_DIR: Path = Field(
        default=Path("data/documents"),
        description="Local directory for NGO verification documents"
    )
    MAX_UPLOAD_SIZE_MB: float = Field(default=10.0, description="Max upload size in MB")
    MAX_DOCUMENT_SIZE_MB: float = Field(default=10.0, description="Max document upload size in MB")
    MAX_UPLOAD_BATCH_SIZE: int = Field(default=10, description="Maximum number of files per evidence batch upload")
    PRESIGNED_URL_EXPIRATION_SECONDS: int = Field(default=3600, ge=60, le=86400, description="Expiration time for presigned storage URLs")
    ALLOWED_MIME_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "application/pdf"],
        description="Permitted upload MIME types"
    )
    ALLOWED_DOCUMENT_MIME_TYPES: List[str] = Field(
        default=["application/pdf", "image/jpeg", "image/png"],
        description="Permitted organization onboarding document MIME types"
    )
    DEFAULT_PAGE_SIZE: int = Field(default=20, description="Default pagination page size")
    MAX_PAGE_SIZE: int = Field(default=100, description="Maximum pagination page size")
    
    # S3 / MinIO Configuration (Phase 5)
    S3_ENDPOINT_URL: Optional[str] = Field(default=None, description="S3 / MinIO custom endpoint")
    S3_ACCESS_KEY: Optional[str] = Field(default=None, description="S3 Access Key")
    S3_SECRET_KEY: Optional[str] = Field(default=None, description="S3 Secret Key")
    S3_BUCKET_NAME: str = Field(default="charitychain-evidence", description="S3 Bucket")
    S3_REGION: str = Field(default="us-east-1", description="S3 Region")

    # 5. AI & Computer Vision Verification Thresholds (Phase 6)
    YOLO_MODEL_PATH: str = Field(default="yolov8n.pt", description="YOLO weights file path")
    YOLO_EXPECTED_HASH: Optional[str] = Field(
        default="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        description="Expected SHA-256 integrity hash for pinned YOLO model artifact"
    )
    YOLO_CONFIDENCE_THRESHOLD: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Per-detection confidence threshold for YOLOv8"
    )
    YOLO_DEVICE: str = Field(default="cpu", description="Execution device for YOLO inference (cpu, cuda)")
    VERIFICATION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum overall confidence score for milestone passing"
    )
    TAMPER_ELA_THRESHOLD: float = Field(
        default=12.0,
        description="Error Level Analysis mean deviation threshold"
    )
    PHASH_DISTANCE_THRESHOLD: int = Field(
        default=10,
        description="pHash Hamming distance threshold for before/after progression"
    )
    SSIM_SIMILARITY_THRESHOLD: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Structural Similarity Index (SSIM) threshold for image comparison"
    )
    GPS_RADIUS_METERS: float = Field(
        default=1000.0,
        description="Geofence proximity radius in meters for evidence verification"
    )
    MIN_DETECTIONS: int = Field(default=1, description="Minimum relevant detected objects")
    VERIFICATION_PIPELINE_VERSION: str = Field(default="v6.0.0", description="Current AI verification pipeline version")
    AI_PROVIDER: str = Field(default="yolo_forensics", description="Active AI verification provider identifier")
    CALIBRATION_VERSION: str = Field(default="heuristic_v1_unvalidated", description="Active calibration methodology version")
    CALIBRATION_STATUS: str = Field(default="UNVALIDATED", description="Calibration statistical status (UNVALIDATED, EMPIRICALLY_CALIBRATED)")
    MAX_IMAGE_PIXELS: int = Field(default=25000000, description="Maximum allowed decoded image pixels to prevent decompression bombs")
    MAX_PDF_PAGES: int = Field(default=10, description="Maximum permitted pages to render for PDF evidence verification")
    DIAGNOSTIC_ARTIFACTS_DIR: str = Field(default="storage/artifacts", description="Filesystem directory for diagnostic visualizations")

    # LLM Explainability Settings (Optional)
    LLM_PROVIDER: Optional[str] = Field(default="none", description="Optional LLM explanation provider ('none', 'openai', 'gemini')")
    LLM_MODEL_NAME: str = Field(default="gpt-4o-mini", description="Model name for LLM explanations")
    LLM_TIMEOUT_SECONDS: float = Field(default=20.0, description="Timeout for external LLM API requests in seconds")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API Key")

    # Background Verification Job Settings
    VERIFICATION_JOB_MAX_ATTEMPTS: int = Field(default=3, description="Maximum retry attempts for failed verification jobs")
    VERIFICATION_JOB_TIMEOUT_SECONDS: int = Field(default=120, description="Processing timeout for verification jobs in seconds")

    # 6. Payment & Financial Contracts (Phase 4 PayPal REST API)
    PAYPAL_MODE: Literal["sandbox", "live"] = Field(default="sandbox", description="PayPal mode")
    PAYPAL_CLIENT_ID: Optional[str] = Field(default=None, description="PayPal Client ID")
    PAYPAL_CLIENT_SECRET: Optional[str] = Field(default=None, description="PayPal Secret")
    PAYPAL_WEBHOOK_ID: Optional[str] = Field(default=None, description="PayPal Webhook ID")
    PAYPAL_API_BASE_URL: Optional[str] = Field(default=None, description="Custom PayPal API Base URL override")
    PAYPAL_CONNECT_TIMEOUT: float = Field(default=10.0, description="PayPal connection timeout in seconds")
    PAYPAL_READ_TIMEOUT: float = Field(default=30.0, description="PayPal read timeout in seconds")
    PAYPAL_RETRY_LIMIT: int = Field(default=3, description="PayPal API max retry attempts for transient errors")
    PAYPAL_WEBHOOK_VERIFY_ENABLED: bool = Field(default=True, description="Enforce cryptographic verification of webhooks")
    MIN_DONATION_AMOUNT: int = Field(default=100, description="Minimum donation amount in minor units (e.g. 100 cents = $1.00)")
    MAX_DONATION_AMOUNT: int = Field(default=100000000, description="Maximum donation amount in minor units (e.g. 100,000,000 cents = $1,000,000)")
    PAYMENT_ORDER_EXPIRY_MINUTES: int = Field(default=60, description="PayPal order expiration window in minutes")
    WEBHOOK_PROCESSING_RETRY_LIMIT: int = Field(default=3, description="Max retries for webhook processing")
    OAUTH_TOKEN_CACHE_TTL_SECONDS: int = Field(default=300, description="Safety margin buffer for OAuth token caching in seconds")
    FRONTEND_RETURN_URL: str = Field(default="http://localhost:8501/?page=donation_result", description="Default checkout return URL")
    FRONTEND_CANCEL_URL: str = Field(default="http://localhost:8501/?page=donation_cancel", description="Default checkout cancel URL")
    DEFAULT_CURRENCY: str = Field(default="USD", description="Base ISO 4217 Currency Code")
    SUPPORTED_CURRENCIES: List[str] = Field(
        default=["USD", "EUR", "GBP", "INR", "ETH"],
        description="List of supported payment currencies"
    )

    # 7. Background Processing & Cache (Redis & Celery)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # 8. Logging & Observability
    LOG_LEVEL: str = Field(default="INFO", description="Application log level")
    LOG_FORMAT: Literal["json", "text"] = Field(default="json", description="Log format")

    # 7. Human Review & Decision Workflows (Phase 7)
    REVIEW_REQUIRED_APPROVALS: int = Field(default=2, ge=1, le=5, description="Number of independent approvals required for milestone sign-off")
    REVIEW_ASSIGNMENT_POLICY: str = Field(default="DUAL_INDEPENDENT", description="Reviewer assignment policy")
    REVIEW_MAX_OPEN_TASKS_PER_REVIEWER: int = Field(default=5, ge=1, description="Maximum active open reviews per reviewer")
    REVIEW_TASK_DUE_HOURS: int = Field(default=72, ge=1, description="Hours until reviewer task is considered overdue")
    REVIEW_SECOND_REVIEW_REQUIRED: bool = Field(default=True, description="Enforce dual-key second review before final milestone approval")
    REVIEW_APPEAL_WINDOW_HOURS: int = Field(default=168, ge=1, description="Appeal submission window in hours (7 days)")
    REVIEW_MAX_APPEALS_PER_CASE: int = Field(default=1, ge=1, description="Maximum formal appeals permitted per review case")
    REVIEW_REVISION_DUE_HOURS: int = Field(default=336, ge=1, description="Hours granted to NGO for revision response (14 days)")
    REVIEW_MAX_REASSIGNMENTS: int = Field(default=3, ge=0, description="Maximum reassignments permitted per task")
    REVIEW_REQUIRE_COMPLETED_AI_REPORT: bool = Field(default=True, description="Require Phase 6 AI verification report before human review assignment")
    REVIEW_REQUIRE_EVIDENCE_INTEGRITY: bool = Field(default=True, description="Require SHA-256 evidence integrity check pass")
    REVIEW_POLICY_VERSION: str = Field(default="v7.0.0", description="Active human review policy version")

    # 9. Payouts & Financial Reconciliation (Phase 8)
    PAYOUT_PROVIDER: Literal["paypal", "mock"] = Field(default="paypal", description="Outbound payout provider")
    PAYPAL_PAYOUTS_WEBHOOK_ID: Optional[str] = Field(default=None, description="PayPal Payouts Webhook ID")
    ENCRYPTION_KEY: Optional[str] = Field(default=None, description="Fernet encryption key for sensitive recipient data at rest")
    DISBURSEMENT_MAX_RETRY_COUNT: int = Field(default=3, description="Maximum retry count for failed disbursements")
    DISBURSEMENT_RETRY_DELAY_SECONDS: int = Field(default=300, description="Delay in seconds before retrying failed disbursement")
    RECONCILIATION_DEFAULT_WINDOW_HOURS: int = Field(default=24, description="Default lookback window for financial reconciliation in hours")

    # 10. Proof-of-Impact Ledger & Certificates (Phase 9)
    ISSUER_ID: str = Field(default="charitychain-core-v1", description="Identifier of the certificate issuer")
    ISSUER_NAME: str = Field(default="CharityChain Platform", description="Display name of the certificate issuer")
    ISSUER_PUBLIC_URL: str = Field(default="http://localhost:8000", description="Public base URL for certificate verification endpoints")
    FRONTEND_VERIFY_URL: str = Field(default="http://localhost:8501/?page=verify_certificate", description="Frontend URL for QR code scanning")
    SIGNING_ALGORITHM: str = Field(default="Ed25519", description="Digital signature algorithm")
    SIGNING_ACTIVE_KEY_ID: str = Field(default="key-charitychain-2026-v1", description="Identifier of the active signing key")
    SIGNING_PRIVATE_KEY_B64: Optional[str] = Field(default=None, description="Base64 encoded Ed25519 private key (optional env var)")
    SIGNING_KEY_DIR: str = Field(default="data/keys", description="Directory for local storage of signing keys in dev")
    CERTIFICATE_SCHEMA_VERSION: str = Field(default="v9.0.0", description="Schema version of issued impact certificates")
    CERTIFICATE_ELIGIBILITY_POLICY_VERSION: str = Field(default="v9.0.0", description="Policy version for certificate eligibility")
    CERTIFICATE_PDF_STORAGE_DIR: str = Field(default="data/certificates/pdf", description="Directory for storing certificate PDFs")
    IMPACT_LEDGER_SCHEMA_VERSION: str = Field(default="v9.0.0", description="Schema version of impact ledger events")
    IMPACT_LEDGER_AUTO_CHECKPOINT_INTERVAL: int = Field(default=50, description="Event interval for generating signed checkpoints")


    @model_validator(mode="after")
    def validate_production_security(self):
        """Enforces that production deployments cannot use development secrets."""
        if self.ENVIRONMENT == "production":
            if "insecure" in self.SECRET_KEY.lower() or len(self.SECRET_KEY) < 32:
                raise ValueError("Insecure default SECRET_KEY detected in production environment! A strong key >= 32 characters is required.")
            if self.DEBUG:
                raise ValueError("DEBUG mode must be disabled (False) in production!")
            if self.STORAGE_PROVIDER in ("s3", "minio"):
                if not self.S3_ACCESS_KEY or not self.S3_SECRET_KEY:
                    raise ValueError("S3_ACCESS_KEY and S3_SECRET_KEY must be provided for S3/MinIO storage in production!")
            if not self.REVIEW_SECOND_REVIEW_REQUIRED or self.REVIEW_REQUIRED_APPROVALS < 2:
                raise ValueError("Dual-key approval cannot be disabled in production! REVIEW_SECOND_REVIEW_REQUIRED=True and REVIEW_REQUIRED_APPROVALS>=2 required.")
        return self


# Global singleton settings instance
settings = Settings()

