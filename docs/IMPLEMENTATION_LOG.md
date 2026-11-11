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
