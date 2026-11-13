# 🌍 CharityChain — AI-Powered Charitable Giving & Impact Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.4+-black.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.3+-38bdf8.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/tests-438%2F438%20passing-brightgreen.svg)](docs/PHASE_14_TEST_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Give with Evidence. Release with Confidence.**

CharityChain is an AI-powered charitable giving and impact verification platform that connects donor contributions directly with verifiable, real-world evidence of completed milestones.

Traditional giving platforms often lack transparency after donations are submitted. NGOs struggle to demonstrate accountability, while donors have limited visibility into whether promised humanitarian work was completed. CharityChain bridges this gap by unifying donations, milestone tracking, multi-algorithm AI forensic verification, human review governance, and verifiable Proof-of-Impact records in one transparent ecosystem.

---

## 🎯 The Core Principle

> **"AI analyzes. Evidence supports. Humans decide. Payment systems execute. CharityChain records."**

- **AI Analyzes**: Computer vision models (YOLOv8, Error Level Analysis, pHash deduplication, SSIM, EXIF GPS extraction) analyze submitted evidence for forensic consistency.
- **Evidence Supports**: NGOs upload verifiable photographs and documentation as proof of milestone delivery.
- **Humans Decide**: Authorized human reviewers inspect AI forensic signals and sign off on approvals, rejections, or revision requests. **AI never independently approves financial releases.**
- **Payment Systems Execute**: Platform explicitly distinguishes between payment order authorization, payment capture, donor refunds, and NGO disbursements (via PayPal). Standard authorization is never misrepresented as smart contract escrow.
- **CharityChain Records**: Every financial event, milestone transition, and verification decision is permanently committed to an append-only, auditable Impact Ledger.

---

## 🏗️ System Architecture

CharityChain is architected as a **Modular Monolith** supporting a decoupled presentation layer:

```
                            ┌────────────────────────────┐
                            │    Presentation Tier       │
                            │  Streamlit UI (Baseline)   │
                            │  Next.js 14+ (Roadmap)     │
                            └─────────────┬──────────────┘
                                          │ REST API / HTTP
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Modular Monolith Backend                           │
│                                                                                 │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌─────────────────────┐  │
│  │   app.core            │  │   app.db & models     │  │   app.integrations  │  │
│  │  - Pydantic Settings  │  │  - Users & Orgs       │  │  - Payments (PayPal)│  │
│  │  - Money Value Object │  │  - Projects & Milest. │  │  - AI Verifier Suite│  │
│  │  - State Machines     │  │  - Financial Ledger   │  │  - Blob Storage     │  │
│  │  - Structured Logging │  │  - Evidence & Reports │  │  - Notifications    │  │
│  │  - Custom Exceptions  │  │  - Certificates &Audit│  │                     │  │
│  └───────────────────────┘  └───────────────────────┘  └─────────────────────┘  │
│                                          │                                      │
│                                          ▼                                      │
│                             SQLAlchemy 2.0 / Alembic                            │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
             PostgreSQL 16 (Relational)                 Redis 7 (Cache/Queue)
             TinyDB (Local Prototyping)                 Local / S3 Blob Storage
```

---

## 🚀 Key Modules & Capabilities

| Module | Purpose & Implementation |
|---|---|
| 🤖 **AI Forensic Verifier** | Multi-algorithm verification suite: YOLOv8 object detection, Error Level Analysis (ELA) for compression anomalies, perceptual hashing (pHash) for image deduplication, SSIM before/after comparison, and true EXIF GPS extraction with Haversine distance validation. Returns structured statuses (`PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`). |
| 💰 **Financial Engine** | Strict `Money` Value Object storing amounts as 64-bit integer minor units (cents) with ISO 4217 currency tracking and Banker's Rounding (`ROUND_HALF_EVEN`). Zero floating-point rounding errors. |
| 🛡️ **State Machine Governance** | Deterministic lifecycles for Organizations, Projects, Milestones, Payment Orders, Captures, Disbursements, and Evidence with role-based access control (`DONOR`, `NGO_ADMIN`, `REVIEWER`, `SUPER_ADMIN`). |
| 🪪 **Proof-of-Impact Certificates** | Verifiable digital certificates awarded to donors containing unique certificate numbers, milestone hashes, and SHA-256 cryptographic digests. |
| 📊 **Impact Analytics** | Real-time project funding progress, verification success rates, milestone distribution, and geographic project mapping. |
| 🔒 **Security & Hardening** | Magic-bytes MIME validation, file size limits (10MB), rate limiting per user, correlation ID tracking (`X-Correlation-ID`), and automatic secret redaction in logs. |

---

## 💻 Quick Start & Running Locally

### 1️⃣ Clone & Set Up Environment

```bash
git clone https://github.com/MeenalSinha/Charity-Chain.git
cd Charity-Chain

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install runtime and dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2️⃣ Configure Environment Variables

Copy the provided template to create your `.env` file (safe development defaults are included):

```bash
cp .env.example .env
```

### 3️⃣ Run Next.js Production Web Application (Primary Frontend)

```bash
cd frontend
pnpm install
pnpm dev
```
Open your browser at **http://localhost:3000/**.

### 4️⃣ Run Streamlit Interactive UI (Auxiliary Fallback)

```bash
streamlit run app_ui.py
```
Open your browser at **http://localhost:8501/**.

### 5️⃣ Run FastAPI Modular Backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```
- Interactive Swagger UI: **http://localhost:8000/docs**
- Health Check: **http://localhost:8000/api/v1/health**

### 6️⃣ Run Multi-Container Topology (Docker Compose)

```bash
docker compose up -d
```
Starts PostgreSQL 16 on port `5432`, Redis 7 on port `6379`, FastAPI Backend on port `8000`, and Next.js Frontend on port `3000`.

---

## 🧪 Automated Testing

CharityChain includes a comprehensive test suite across unit, integration, functional, and security domains:

```bash
# Run all tests
pytest -v tests/
```

**Test Results: 337 passed out of 337 tests across 102 test files (100% pass rate):**
- **Unit Tests (100% passing)**: Domain events, SSRF protection, HMAC signing, email templates, outbox backoff, fund allocation, ledger balancing, AI forensics, state machines.
- **Integration Tests (100% passing)**: Notification worker outbox loops, in-app notification center, webhook subscriptions, milestone payouts, financial reconciliation, payment captures, refunds.
- **Security Tests (100% passing)**: Multi-tenant IDOR notification isolation, webhook cross-tenant defense, Maker-Checker segregation, ledger immutability, encryption at rest.
- **Functional E2E Tests (100% passing)**: Multi-channel donation receipts, certificate issuance -> HMAC webhook dispatch, disbursement flows, review workflows.

---

## 📁 Repository Structure

```
Charity-Chain/
├── app_ui.py                   # Streamlit frontend application
├── charity_tracker.py          # Core tracker domain engine (stabilized baseline)
├── docker-compose.yml          # Local container topology (PostgreSQL, Redis, API)
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development, testing, and backend dependencies
├── pytest.ini                  # Pytest configuration
├── .env.example                # Documented configuration template (safe placeholders)
├── .gitignore                  # Git ignore rules (secrets, logs, caches, dbs)
│
├── backend/                    # Phase 1 Modular Monolith Backend
│   ├── Dockerfile              # Multi-stage production container
│   ├── pyproject.toml          # Python packaging
│   ├── alembic.ini             # Alembic migration config
│   ├── alembic/                # Relational schema migrations
│   └── app/
│       ├── main.py             # FastAPI entry point, CORS, middlewares
│       ├── api/                # REST v1 routers, schemas, envelopes
│       ├── core/               # Config, currency, exceptions, logging, state machines
│       ├── db/                 # Base, session pool, SQLAlchemy domain models
│       └── integrations/       # Port/adapter interfaces (payments, AI, storage, notifications)
│
├── data/                       # Application data directory
│   ├── charity.db              # Baseline TinyDB database (backed up at charity.db.bak)
│   └── evidence/               # Evidence blob storage directory
│
├── docs/                       # Complete Phase 0 & Phase 1 Documentation Suite
│   ├── ARCHITECTURE.md         # Full system architecture & Mermaid diagrams
│   ├── DOMAIN_MODEL.md         # Entity specifications & business invariants
│   ├── DATABASE_DESIGN.md      # PostgreSQL schema, indexes, ER diagram, migration strategy
│   ├── API_DESIGN.md           # REST v1 endpoints, error codes, envelopes
│   ├── CONFIGURATION.md        # Environment variables reference & validation rules
│   ├── STATE_MACHINES.md       # Entity lifecycles & authorized actor matrix
│   ├── INTEGRATION_DESIGN.md   # External provider abstractions & contracts
│   ├── PHASE_0_AUDIT_REPORT.md # Initial codebase audit & gap analysis
│   ├── PHASE_0_COMPLETION_REPORT.md # Phase 0 stabilization verification
│   ├── PHASE_1_COMPLETION_REPORT.md # Phase 1 delivery & gate signoff
│   ├── PHASE_STATUS.md         # Comprehensive task-by-task status ledger
│   ├── TEST_REPORT.md          # Full 50-test execution report
│   ├── IMPLEMENTATION_LOG.md   # Chronological decision & fix log
│   ├── KNOWN_LIMITATIONS.md    # Known limitations & boundary conditions
│   └── DEVELOPMENT_ROADMAP.md  # Multi-phase engineering roadmap (Phases 0–14)
│
└── tests/                      # Automated test suite
    ├── conftest.py             # Shared pytest fixtures
    ├── functional/             # UI & analytics tests
    ├── integration/            # Workflow & FastAPI integration tests
    ├── security/               # Security & input hardening tests
    └── unit/                   # Unit tests (currency, AI, db, state machines, etc.)
```

---

## 📖 Documentation Index

For in-depth architectural and domain specifications, refer to the documents in [`docs/`](docs/):
- [System Architecture Specification](docs/ARCHITECTURE.md)
- [Domain Model Specification](docs/DOMAIN_MODEL.md)
- [Database Design & ER Diagram](docs/DATABASE_DESIGN.md)
- [REST API v1 Specification](docs/API_DESIGN.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [State Machine & Governance Rules](docs/STATE_MACHINES.md)
- [External Integration Design](docs/INTEGRATION_DESIGN.md)
- [Phase 0 Completion Report](docs/PHASE_0_COMPLETION_REPORT.md)
- [Phase 1 Completion Report](docs/PHASE_1_COMPLETION_REPORT.md)
- [Phase 2 Completion Report](docs/PHASE_2_COMPLETION_REPORT.md)
- [Phase 3 Completion Report](docs/PHASE_3_COMPLETION_REPORT.md)
- [Phase 4 Completion Report](docs/PHASE_4_COMPLETION_REPORT.md)
- [Phase 5 Completion Report](docs/PHASE_5_COMPLETION_REPORT.md)
- [Phase 6 Completion Report](docs/PHASE_6_COMPLETION_REPORT.md)
- [Phase 7 Completion Report](docs/PHASE_7_COMPLETION_REPORT.md)
- [Phase 7 Test Report (234 Tests)](docs/PHASE_7_TEST_REPORT.md)
- [Phase 8 Completion Report](docs/PHASE_8_COMPLETION_REPORT.md)
- [Phase 8 Test Report (263 Tests)](docs/PHASE_8_TEST_REPORT.md)
- [Phase 9 Completion Report](docs/PHASE_9_COMPLETION_REPORT.md)
- [Phase 10 Completion Report](docs/PHASE_10_COMPLETION_REPORT.md)
- [Phase 10 Test Report (337 Tests)](docs/PHASE_10_TEST_REPORT.md)
- [Phase 11 Completion Report](docs/PHASE_11_COMPLETION_REPORT.md)
- [Phase 11 Test Report (358 Tests)](docs/PHASE_11_TEST_REPORT.md)
- [Phase 11 Database Changes](docs/PHASE_11_DATABASE_CHANGES.md)
- [Analytics & Transparency Architecture](docs/ANALYTICS_ARCHITECTURE.md)
- [Canonical Metric Definitions](docs/METRIC_DEFINITIONS.md)
- [NGO Analytics & Performance](docs/NGO_ANALYTICS.md)
- [Donor Portfolio Analytics](docs/DONOR_PORTFOLIO_ANALYTICS.md)
- [Public Transparency & Open Governance](docs/PUBLIC_TRANSPARENCY.md)
- [Geographic Impact Mapping & Location Privacy](docs/GEOGRAPHIC_IMPACT_MAPPING.md)
- [Financial Analytics & Reconciliation Methodology](docs/FINANCIAL_ANALYTICS_METHODOLOGY.md)
- [Analytics Privacy & Security](docs/ANALYTICS_PRIVACY_AND_SECURITY.md)
- [Analytics API Reference](docs/ANALYTICS_API_REFERENCE.md)
- [Analytics Data Exports](docs/ANALYTICS_EXPORTS.md)
- [Phase 9 Test Report (306 Tests)](docs/PHASE_9_TEST_REPORT.md)
- [Proof-of-Impact Certificates](docs/PROOF_OF_IMPACT_CERTIFICATES.md)
- [Certificate Cryptography & Verification](docs/CERTIFICATE_CRYPTOGRAPHY.md)
- [Impact Ledger Architecture](docs/IMPACT_LEDGER_ARCHITECTURE.md)
- [Certificate Privacy & PII Protection](docs/CERTIFICATE_PRIVACY.md)
- [Public Certificate & Ledger API](docs/CERTIFICATE_VERIFICATION_API.md)
- [Issuer Key Management & Security](docs/CERTIFICATE_KEY_MANAGEMENT.md)
- [Phase 9 Database Changes](docs/PHASE_9_DATABASE_CHANGES.md)
- [Financial Architecture & Systems](docs/FINANCIAL_ARCHITECTURE.md)
- [Disbursement Workflow & Eligibility](docs/DISBURSEMENT_WORKFLOW.md)
- [Payout Provider Integration](docs/PAYOUT_PROVIDER_INTEGRATION.md)
- [Payout Recipient Management](docs/PAYOUT_RECIPIENT_MANAGEMENT.md)
- [Double-Entry Ledger Specification](docs/DOUBLE_ENTRY_LEDGER.md)
- [Fund Allocation Specification](docs/FUND_ALLOCATION.md)
- [Automated Financial Reconciliation](docs/FINANCIAL_RECONCILIATION.md)
- [Financial Security & Controls](docs/FINANCIAL_SECURITY.md)
- [Phase 8 Database Changes](docs/PHASE_8_DATABASE_CHANGES.md)
- [Human Review Workflows Specification](docs/REVIEW_WORKFLOW.md)
- [Reviewer Governance & COI Rules](docs/REVIEWER_GOVERNANCE.md)
- [Dual-Key Approval Architecture](docs/DUAL_KEY_APPROVAL.md)
- [Revision & Resubmission Workflow](docs/REVISION_AND_RESUBMISSION.md)
- [Formal Dispute Resolution](docs/DISPUTE_RESOLUTION.md)
- [Review Audit Trail](docs/REVIEW_AUDIT_TRAIL.md)
- [Review API Documentation](docs/REVIEW_API.md)

---

## 🚀 Deployment

CharityChain is designed for Google Cloud Run with Terraform IaC.

### Quick Start (Local Development)
```bash
# Clone and start development environment
git clone https://github.com/MeenalSinha/Charity-Chain.git
cd Charity-Chain
cp .env.example .env
docker compose up -d
```

### Production Deployment
```bash
# See docs/PHASE_14_DEPLOYMENT_ARCHITECTURE_DECISION.md
# Requires GCP project + environment variables from .env.production.example
cp .env.production.example .env
# Fill in all required secrets before proceeding
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Infrastructure as Code
```bash
cd infra/terraform/environments/production
cp terraform.tfvars.example terraform.tfvars
# Fill in project_id, images, alert_email
terraform init
terraform plan
terraform apply
```

### Key Deployment Documents
- [Deployment Architecture Decision](docs/PHASE_14_DEPLOYMENT_ARCHITECTURE_DECISION.md)
- [Backup Strategy](docs/PHASE_14_BACKUP_STRATEGY.md)
- [Disaster Recovery Drills](docs/PHASE_14_DISASTER_RECOVERY_DRILLS.md)
- [Database Migration Strategy](docs/PHASE_14_DATABASE_MIGRATION_STRATEGY.md)
- [Operational Runbooks](docs/PHASE_14_OPERATIONAL_RUNBOOKS.md)
- [Capacity Planning](docs/PHASE_14_CAPACITY_PLANNING.md)

---

## 👥 User Roles & Governance

1. **Donors**: Discover verified charitable projects, contribute via PayPal, track milestone progress, and receive cryptographic Proof-of-Impact certificates.
2. **NGOs**: Create projects and milestones, submit photographs and documentation as evidence of completed work, and receive disbursements upon verification.
3. **Reviewers / Verifiers**: Inspect AI forensic signals, review submitted evidence, approve/reject milestone deliverables, and trigger payment capture workflows.
4. **Super Administrators**: Platform governance, organization onboarding, tax ID validation, and compliance audits.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
