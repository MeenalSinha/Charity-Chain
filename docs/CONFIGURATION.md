# CharityChain — Configuration Specification
**Phase 1 Deliverable — Zero-Hardcoding Architecture, Environment Variables & Validation Rules**

---

## 1. Configuration Philosophy & Architecture

CharityChain enforces a strict **Zero-Hardcoding Policy**:
- **Pydantic Settings**: All configuration parameters are declared and validated using Pydantic Settings (`backend/app/core/config.py`).
- **Strict Environment Segregation**: Behavior adjusts deterministically according to `ENVIRONMENT` (`development`, `staging`, `production`, `testing`).
- **Production Guardrails**: In `production`, default development keys (such as insecure `SECRET_KEY` strings) trigger immediate startup failure.
- **Secret Sanitization**: Secrets and credentials are automatically scrubbed from application logs, error traces, and diagnostic endpoints.

---

## 2. Environment Variables Catalog

### 2.1 Application & Server Settings

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `APP_NAME` | str | Optional | `CharityChain` | Non-empty | Application display name |
| `APP_VERSION` | str | Optional | `2.0.0` | SemVer format | API semantic version string |
| `ENVIRONMENT` | str | Optional | `development` | In `[development, staging, production, testing]` | Active runtime environment |
| `DEBUG` | bool | Optional | `True` | Automatically False in `production` | Enables verbose debug mode & Swagger UI |
| `API_PREFIX` | str | Optional | `/api/v1` | Must start with `/` | Global API version route prefix |
| `HOST` | str | Optional | `0.0.0.0` | Valid IP | Server binding interface |
| `PORT` | int | Optional | `8000` | 1024 - 65535 | HTTP listener port |
| `CORS_ORIGINS` | List[str] | Optional | `["http://localhost:3000", "http://localhost:8501"]` | Valid URLs | Permitted Cross-Origin Resource Sharing origins |

### 2.2 Security & Cryptography

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `SECRET_KEY` | str | **Required in Prod** | `dev-insecure-secret-key-change-in-production-charity-chain` | Min 32 chars in production | Master cryptographic salt for JWT signing |
| `ALGORITHM` | str | Optional | `HS256` | Valid JWT algorithm | Signature algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | int | Optional | `60` | > 0 | JWT access token validity duration |
| `RATE_LIMIT_UPLOADS_PER_HOUR` | int | Optional | `10` | 1 - 1000 | Maximum evidence uploads per user per hour |

### 2.3 Database & Cache

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `DATABASE_URL` | str | Optional | `sqlite:///./data/charity_v2.db` | Valid DB URI (PostgreSQL in Prod) | Primary relational database connection URI |
| `DB_POOL_SIZE` | int | Optional | `5` | 1 - 50 | SQLAlchemy connection pool size |
| `DB_MAX_OVERFLOW` | int | Optional | `10` | 0 - 50 | Maximum overflow connections beyond pool size |
| `DB_TIMEOUT` | int | Optional | `30` | 5 - 120 | Database query execution timeout in seconds |
| `REDIS_URL` | str | Optional | `redis://localhost:6379/0` | Valid Redis URI | Redis cache and task queue broker URI |

### 2.4 Evidence & Storage

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `STORAGE_BACKEND` | str | Optional | `local` | `local` or `s3` | File storage provider adapter |
| `LOCAL_STORAGE_DIR` | str | Optional | `data/evidence` | Valid directory path | Path for local evidence blob storage |
| `MAX_UPLOAD_SIZE_MB` | float | Optional | `10.0` | 0.1 - 50.0 | Maximum allowed file upload size |
| `ALLOWED_MIME_TYPES` | List[str] | Optional | `["image/jpeg", "image/png"]` | MIME types | Strictly allowed evidence MIME types |

### 2.5 AI & Forensic Thresholds

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `YOLO_MODEL_PATH` | str | Optional | `models/yolov8n.pt` | File path | Path to local YOLOv8 neural network weights |
| `ELA_TAMPER_THRESHOLD` | float | Optional | `12.0` | 1.0 - 50.0 | Error Level Analysis mean difference threshold |
| `PHASH_DISTANCE_THRESHOLD` | int | Optional | `10` | 0 - 64 | Hamming distance threshold for image duplicate detection |
| `SSIM_THRESHOLD` | float | Optional | `0.4` | 0.0 - 1.0 | Structural similarity threshold for scene matching |
| `GPS_RADIUS_METERS` | float | Optional | `1000.0` | 10.0 - 50000.0 | Maximum allowed distance from project site |
| `VERIFICATION_CONFIDENCE_THRESHOLD` | float | Optional | `0.65` | 0.0 - 1.0 | Normalized minimum confidence for automated pass indicator |
| `MIN_DETECTIONS` | int | Optional | `1` | >= 0 | Minimum required object detections |

### 2.6 External Integrations (PayPal & Notifications)

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `PAYPAL_MODE` | str | Optional | `sandbox` | `sandbox` or `live` | PayPal gateway operating environment |
| `PAYPAL_CLIENT_ID` | str | Optional | `DEMO_MODE` | Non-empty for live | PayPal REST API client ID |
| `PAYPAL_CLIENT_SECRET` | str | Optional | `DEMO_MODE` | Non-empty for live | PayPal REST API secret |
| `PAYPAL_WEBHOOK_ID` | str | Optional | `DEMO_MODE` | Non-empty for live | PayPal webhook ID for event signature verification |
| `SMTP_HOST` | str | Optional | `localhost` | Valid host | Outbound email SMTP host |
| `SMTP_PORT` | int | Optional | `587` | 1 - 65535 | Outbound email SMTP port |
| `SMTP_USER` | str | Optional | `None` | Optional | SMTP username |
| `SMTP_PASSWORD` | str | Optional | `None` | Optional | SMTP password |
| `EMAIL_FROM` | str | Optional | `noreply@charitychain.org` | Valid email | Default sender email address |

### 2.7 Observability & Logging

| Variable Name | Type | Req/Opt | Default Value | Validation Rules | Description |
|---|---|---|---|---|---|
| `LOG_LEVEL` | str | Optional | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Logging severity threshold |
| `LOG_FORMAT` | str | Optional | `json` | `json` or `text` | Logging structure format |

