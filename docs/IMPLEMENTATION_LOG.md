# CharityChain — Implementation Log

## 2026-11-11: Phase 0 Stabilization Initiated

### Discovery & Root Cause Analysis
1. **Critical TinyDB Truthiness Bug**:
   - `TinyDB` evaluates `__len__` on the `_default` table. When data is stored in named tables (`milestones`, `donations`, etc.), `bool(self.db)` evaluated to `False`.
   - This caused `self.db` checks across `charity_tracker.py` to silently fail or return empty/default records, triggering widespread debugging scripts in `tools/`.
   - **Fix**: Replaced all `if self.db:` checks with `if self.db is not None:` and restored deterministic `_init_tables()` method.
2. **Fabricated AI & Computer Vision Results**:
   - When YOLO was unavailable, hardcoded mock detections (`person: 0.85`, `construction: 0.72`) were returned.
   - GPS check contained a hardcoded `distance = 100` mock distance, completely ignoring actual image metadata.
   - When before/after image was omitted, the check was awarded a default `passed=True` score, inflating confidence.
   - **Fix**:
     - Real EXIF extraction with true DMS latitude/longitude parsing.
     - Unavailable YOLO returns empty detections and `UNAVAILABLE` status.
     - GPS calculation uses true Haversine distance if EXIF data is present; otherwise marks `UNAVAILABLE`.
     - Standardized check report statuses: `PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`, `PENDING`.
     - Confidence calculation weights only executed applicable checks.
3. **Misleading IPFS and Blockchain Representations**:
   - Code claimed to mint ERC-721 NFTs on Polygon Mumbai testnet and upload to Web3.Storage, while actually generating local SHA-256 strings in demo mode.
   - **Fix**: Reframed all digital assets as "Proof-of-Impact Certificates" (verifiable local digital records) and "Milestone-Locked Funds" (application-managed release).
4. **Security & Configuration**:
   - `.env` file was tracked in Git repository without `.gitignore` protection.
   - No `.env.example` existed.
   - **Fix**: Created `.gitignore`, generated `.env.example`, untracked `.env` from Git index, verified no active private keys were compromised.
5. **Data Protection**:
   - Backed up `data/charity.db` to `data/charity.db.bak` before any stabilization modifications.


## 2026-11-11: Phase 1 Architecture & Domain Implementation

### Architectural Decisions & Implementations
1. **Modular Monolith Foundation**:
   - Structured `backend/app/` into `core`, `db`, `integrations`, and `api` bounded contexts.
   - Built FastAPI application factory with CORS middleware, correlation ID tracking, and global exception handlers.
2. **Zero-Hardcoding Configuration (Pydantic Settings)**:
   - Implemented typed `Settings` class in `backend/app/core/config.py`.
   - Strictly enforced environment separation (`development`, `staging`, `production`, `testing`).
   - Added production guardrails preventing default development keys from starting up in production.
3. **Currency Precision & Financial Engine**:
   - Implemented `Money` value object (`backend/app/core/currency.py`) operating strictly in integer minor units (cents).
   - Enforced Banker's Rounding (`ROUND_HALF_EVEN`) using Python's `decimal` module, eliminating floating-point rounding errors.
4. **Deterministic State Machines**:
   - Implemented `StateMachineValidator` in `backend/app/core/state_machines.py`.
   - Defined valid transitions and role authorizations for Organizations, Projects, Milestones, Payments, and Evidence.
   - Prohibited invalid transitions with typed `InvalidStateTransitionException`.
5. **Relational Database & Migrations**:
   - Implemented declarative models in SQLAlchemy 2.0 (`backend/app/db/models/`).
   - Primary keys use UUID v4 strings, financial amounts use `BigInteger` cents, timestamps are in UTC.
   - Configured Alembic migration environment with initial migration `001_initial_schema.py`.
6. **Provider Abstractions (Ports & Adapters)**:
   - Established abstract interfaces for Payments (`PaymentProviderInterface`), AI Verifiers (`AIProviderInterface`), Storage (`StorageProviderInterface`), and Notifications (`NotificationProviderInterface`).
7. **Containerization Topology**:
   - Created `docker-compose.yml` defining services for PostgreSQL 16, Redis 7, and FastAPI Backend.
   - Created multi-stage `backend/Dockerfile` and `backend/pyproject.toml`.
8. **Automated Test Suite Expansion**:
   - Expanded test suite from 29 to 50 automated tests covering currency arithmetic, state machines, database models, Pydantic settings, exception handling, and FastAPI endpoints.
   - Achieved 100% pass rate (50 passed in 6.95s).


## 2026-11-11: Phase 2 Authentication, Identity & Access Control Implemented

### Architectural Decisions & Implementations
1. **Separation of Platform and Organization Roles**:
   - Disentangled the monolithic role model into global `PlatformRole` (`DONOR`, `REVIEWER`, `PLATFORM_ADMIN`) and multi-tenant `OrganizationRole` (`OWNER`, `ADMIN`, `MEMBER`).
   - Maintained `UserRole` as a backward-compatible mapping for existing codebase components.
2. **Argon2id Password Cryptography**:
   - Replaced raw/insecure password handling with standard Argon2id via `argon2-cffi`.
   - Tuned parameters: 64MB memory cost, 3 iterations, 4 parallelism threads, 32-byte hash length, with constant-time verification.
3. **Session & Refresh Token Lifecycle (Compromise Detection)**:
   - Built a persistent `UserSession` model tracking IP, user-agent, creation, and last-active timestamps.
   - Implemented single-use refresh token rotation with a SHA-256 storage hash.
   - Built Token Family Compromise Detection: if a previously-rotated refresh token is submitted, all active sessions belonging to that token family are revoked immediately.
4. **Anti-Enumeration and Brute Force Lockout**:
   - Login, forgot-password, and verification resend endpoints return identical generic timing-safe responses regardless of whether the email exists.
   - Account lockout triggers automatically after 5 consecutive failed attempts, locking the account for 15 minutes.
5. **Multi-Tenancy & Last Owner Protection**:
   - Scoped organization membership via `OrganizationMembership` and `OrganizationInvitation`.
   - Prohibited deleting or demoting the last remaining active `OWNER` of an organization.
6. **Administrative CLI Bootstrapping**:
   - Implemented `backend/app/cli.py` providing `create-platform-admin` command with email normalization, password confirmation, and secure DB bootstrapping.
7. **Streamlit Transitional UI Security**:
   - Retired the insecure sidebar select box that allowed arbitrary role-switching without authentication.
   - Added real Sign In and Register tabs connected to `AuthService` and persistent database session factory.
8. **Automated Verification Expansion**:
   - Expanded test suite from 50 to 78 tests across unit, integration, functional, and security tiers, achieving 100% pass rate (78 passed in 5.01s).


## 2026-11-11: Phase 3 Organization Onboarding, NGO Verification & Project Management Implemented

### Architectural Decisions & Implementations
1. **Multi-Tenant Legal Organization Onboarding**:
   - Expanded `Organization` model with compliance fields: `legal_name`, `registration_authority`, `country`, `registered_address`, `mission_statement`, `focus_areas`.
   - Designed atomic registration flow that provisions the organization and binds the creator as `OWNER`.
   - Created `OrganizationDocument` model for tracking verification filings with SHA-256 hashes and MIME verification.
2. **High-Assurance Document Storage (`LocalStorageProvider`)**:
   - Implemented binary magic byte inspection for `application/pdf`, `image/png`, and `image/jpeg`.
   - Enforced 10MB maximum file size ceiling.
   - Implemented path traversal prevention rejecting payloads targeting parent directories or system paths.
3. **Conflict of Interest Prevention**:
   - Reviewers holding an active role in an organization are strictly blocked from reviewing or deciding on that organization or its projects (`ForbiddenException`).
4. **Multi-Milestone Project Management**:
   - Replaced empty API placeholders in `backend/app/api/v1/projects.py` and `backend/app/api/v1/milestones.py` with full domain services (`ProjectService`).
   - Implemented database constraint `uq_milestone_project_seq` and enforced contiguous `1..N` sequence numbering.
   - Enforced budget allocation invariants: total milestone funding targets cannot exceed the project target ceiling.
5. **13-Rule Pre-Submission Live Validation Engine**:
   - Built pre-submission checklist in `ProjectService.validate_project_submission` covering organization verification, title/summary lengths, budget matching, sequence contiguity, deliverables, and GPS bounds.
   - Exposed `/api/v1/projects/{id}/validation` for real-time diagnostics before submission.
6. **Public Project Discovery Engine**:
   - Implemented `GET /api/v1/projects` with strict multi-factor visibility filters: `Project.status == ACTIVE`, `Project.public_visibility == PUBLIC`, `Organization.status == VERIFIED`, `Project.is_archived == False`.
   - Added support for keyword search, category filtering, country filtering, and pagination.
7. **Streamlit UI Expansion**:
   - Added dedicated pages in `app_ui.py`: Organization Onboarding, Project Management (with live 13-rule checklist), Reviewer Queue, and Discover Projects.
   - Maintained full backward compatibility with original Streamlit routes and TinyDB milestones.
8. **Database Schema & Alembic Migration**:
   - Generated migration `003_phase3_org_and_projects.py` with fully reversible `upgrade()` and `downgrade()` logic.
9. **Comprehensive Automated Testing**:
   - Expanded test suite from 78 to 107 tests across unit, integration, security, and functional layers.
   - 107 passed in 35.35s (100% pass rate).

---

## Phase 4: PayPal Payment Gateway Integration & Donation Processing

**Date**: 2026-11-11  
**Status**: COMPLETED (133/133 tests passing)

### Architectural Decisions & Changes
1. **Payment Provider Abstraction**:
   - Implemented `PaymentProviderInterface` in `backend/app/integrations/payments/base.py`.
   - Created `PayPalPaymentProvider` (`backend/app/integrations/payments/paypal.py`) communicating with PayPal Orders v2 and Payments v2.
   - Built `MockPayPalProvider` (`tests/mocks/paypal_mock.py`) for deterministic test execution.
2. **OAuth 2.0 Token Management**:
   - Acquired client credentials Bearer tokens via `POST /v1/oauth2/token`.
   - Cached in memory with configurable safety buffer `OAUTH_TOKEN_CACHE_TTL_SECONDS`.
   - Synchronized token renewal with `threading.Lock()` to prevent race conditions.
3. **Intent CAPTURE Implementation**:
   - Configured PayPal Orders v2 with `intent=CAPTURE` and purchase units linked to unique donation UUIDs.
   - Enforced immediate capture rather than long-lived authorizations to eliminate authorization fee overhead and 29-day expiration hazards.
4. **Server-Side Validation Engine**:
   - Enforced project archive status, active status, public visibility, NGO verification, milestone status, currency matching, and min/max monetary bounds ($1.00 to $100,000.00).
5. **Idempotent Payment Capture**:
   - Implemented `DonationService.capture_donation_payment()` validating PayPal order state (`APPROVED`/`COMPLETED`) and matching amounts before persisting `PaymentCapture`.
   - Prevented double-capture and double-counting of raised funds on repeated calls.
6. **Cryptographic Webhook Signature Verification**:
   - Implemented `POST /api/v1/webhooks/paypal` calling PayPal's verification API.
   - Enforced event deduplication using the `webhook_events` table (`event_id UNIQUE`).
   - Implemented automatic payment capture on `CHECKOUT.ORDER.APPROVED` for donors who drop off before returning.
7. **Full & Partial Refund Management**:
   - Implemented `DonationService.process_refund()` executing via PayPal Payments v2 Refund API.
   - Enforced remaining refundable balance ceiling (`capture_amount - sum(completed_refunds)`).
   - Decremented target milestone and project raised funding totals in real time.
8. **Financial Data Integrity**:
   - All financial operations executed strictly in integer minor units via `Money` value object.
   - Provided real-time funding summaries (`captures - refunds`).
   - Kept captured funds in platform holding accounts and deferred NGO disbursements to Phase 8.
9. **Streamlit UI Integration**:
   - Enhanced Project Discovery page with milestone contribution selectors and PayPal approval links.
   - Added Donor Dashboard with KPI cards, donation history, and interactive refund request forms.
10. **Database Migration**:
   - Authored Alembic migration `004_phase4_payment_and_donations.py` modifying donations, payment orders, captures, refunds, and creating `webhook_events`.
11. **Comprehensive Automated Testing**:
   - Added 10 new test suites (26 tests). All 133 tests passing with 0 failures, 0 errors.

## Phase 5: Evidence Management System (Secure Uploads, S3/MinIO Storage, Deduplication & Tamper-Evident Records)
- **Status**: COMPLETED
- **Key Changes**:
  - Implemented `StorageProviderInterface` with synchronous methods for zero-deadlock integration with SQLAlchemy.
  - Implemented `LocalStorageProvider` with binary magic-byte inspection (PDF, PNG, JPEG), filename sanitization, and canonical path containment (`target.relative_to(self.base_dir)`).
  - Implemented `S3StorageProvider` via `boto3` supporting AWS S3 and MinIO with short-lived presigned URLs (60s to 86,400s) and user metadata storage.
  - Implemented `get_storage_provider()`, `set_storage_provider()`, and `reset_storage_provider()` factory.
  - Enriched `EvidenceFile` model with `original_filename`, `storage_key`, `storage_provider`, `storage_version_id`, `detected_mime_type`, `version_number`, `parent_evidence_id`, `is_archived`, `archived_at`, `quarantine_reason`.
  - Created `EvidenceAuditEvent` model with deterministic SHA-256 hash chaining.
  - Created and executed Alembic migration `005_phase5_evidence_management` (forward and rollback verified).
  - Implemented `EvidenceService` with multi-file batch uploads, duplicate detection on milestones raising `ConflictException`, multi-versioning, real-time SHA-256 integrity verification, soft-archival, and unbroken audit chain mathematical validation.
  - Mounted REST API endpoints under `/api/v1/milestones/{id}/evidence` and `/api/v1/evidence/...`.
  - Integrated interactive Phase 5 Evidence Engine into `app_ui.py` alongside legacy demo tab.
  - Added 24 new automated tests across 8 suites. Total test suite: 157 / 157 passed (100%) in 68.72s.

## Phase 6: AI Evidence Verification Engine (Computer Vision, Forensics, Geospatial, Progress & XAI)
- **Status**: COMPLETED
- **Key Changes**:
  - Conducted pre-Phase 6 audit and repaired authorization boundaries across evidence endpoints (`_check_evidence_access`), preventing donors from viewing unapproved evidence or private audit logs.
  - Replaced unbounded upload memory buffering with 64KB chunked streaming and magic bytes inspection.
  - Designed strongly typed AI schemas in `backend/app/integrations/ai/schemas.py` and `AIProviderInterface` in `backend/app/integrations/ai/provider.py`.
  - Implemented real YOLOv8 object detection in `backend/app/integrations/ai/yolo_provider.py` with SHA-256 pinned weights check (`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`) and category relevance mapping in `relevance_mapper.py`.
  - Implemented JPEG Error Level Analysis in `backend/app/integrations/ai/ela_analyzer.py` with difference variance calculation and amplified diagnostic heatmap generation; non-JPEGs handled as `NOT_APPLICABLE`.
  - Implemented perceptual hashing (64-bit DCT pHash) and SSIM comparison in `phash_analyzer.py`, `ssim_analyzer.py`, and `progress_comparator.py`; duplicate baseline rejection vs visible physical progression detection.
  - Implemented authentic EXIF metadata extraction and GPS DMS coordinate parsing with Haversine distance evaluation in `exif_analyzer.py` and `gps_analyzer.py`.
  - Implemented safe bounded PDF processing (10 pages / 50MB) in `pdf_processor.py`.
  - Implemented grounded Explainable AI with prompt-injection defense and deterministic rule-based fallback in `llm_explainer.py`.
  - Standardized `calibration_status = "UNVALIDATED"` across all reports and API contracts.
  - Extended database models (`VerificationReport`, `VerificationCheckRecord`, `VerificationJob`) and authored Alembic migration `006_phase6_ai_verification.py` (verified `upgrade`/`downgrade`/`upgrade`).
  - Implemented `VerificationService` with queuing, status tracking, retry handling, sync/async worker execution, and cryptographic audit chaining (`AI_VERIFIED`).
  - Mounted 6 REST endpoints under `/api/v1/verification/` in `backend/app/api/v1/verification.py`.
  - Integrated Phase 6 verification findings panel and diagnostic artifact viewer in Streamlit `app_ui.py`.
  - Wired legacy `EnhancedAIVerifier` in `charity_tracker.py` to delegate directly to modular backend AI engine.
  - Added 40 new automated tests across 9 suites. Total test suite: 197 / 197 passed (100%) in 136.77s.

