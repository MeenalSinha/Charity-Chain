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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="Token validity window")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token validity")

    # 4. Storage & Files
    STORAGE_PROVIDER: Literal["local", "s3", "pinata"] = Field(
        default="local",
        description="Selected file storage backend"
    )
    STORAGE_LOCAL_DIR: Path = Field(
        default=Path("data/evidence"),
        description="Local directory for evidence storage"
    )
    MAX_UPLOAD_SIZE_MB: float = Field(default=10.0, description="Max upload size in MB")
    ALLOWED_MIME_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "application/pdf"],
        description="Permitted upload MIME types"
    )
    
    # S3 / MinIO Configuration (Phase 5)
    S3_ENDPOINT_URL: Optional[str] = Field(default=None, description="S3 / MinIO custom endpoint")
    S3_ACCESS_KEY: Optional[str] = Field(default=None, description="S3 Access Key")
    S3_SECRET_KEY: Optional[str] = Field(default=None, description="S3 Secret Key")
    S3_BUCKET_NAME: str = Field(default="charitychain-evidence", description="S3 Bucket")
    S3_REGION: str = Field(default="us-east-1", description="S3 Region")

    # 5. AI & Computer Vision Verification Thresholds
    YOLO_MODEL_PATH: str = Field(default="yolov8n.pt", description="YOLO weights file path")
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
    GPS_RADIUS_METERS: float = Field(
        default=1000.0,
        description="Geofence proximity radius in meters for evidence verification"
    )
    MIN_DETECTIONS: int = Field(default=1, description="Minimum relevant detected objects")

    # AI API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API Key")

    # 6. Payment & Financial Contracts (Phase 4 PayPal REST API)
    PAYPAL_MODE: Literal["sandbox", "live"] = Field(default="sandbox", description="PayPal mode")
    PAYPAL_CLIENT_ID: Optional[str] = Field(default=None, description="PayPal Client ID")
    PAYPAL_CLIENT_SECRET: Optional[str] = Field(default=None, description="PayPal Secret")
    PAYPAL_WEBHOOK_ID: Optional[str] = Field(default=None, description="PayPal Webhook ID")
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

    @model_validator(mode="after")
    def validate_production_security(self):
        """Enforces that production deployments cannot use development secrets."""
        if self.ENVIRONMENT == "production":
            if "insecure" in self.SECRET_KEY.lower() or len(self.SECRET_KEY) < 32:
                raise ValueError("Insecure default SECRET_KEY detected in production environment! A strong key >= 32 characters is required.")
            if self.DEBUG:
                raise ValueError("DEBUG mode must be disabled (False) in production!")
        return self


# Global singleton settings instance
settings = Settings()

