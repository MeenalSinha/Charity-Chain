# CharityChain Backend Engine (Modular Monolith)

## Overview
The CharityChain backend is built on **FastAPI**, **SQLAlchemy (Async)**, and **Pydantic Settings**.
It provides the core architecture and domain models for evidence-backed giving, fraud detection, and verifiable disbursements.

## Directory Structure
```
backend/
├── app/
│   ├── main.py                   # FastAPI Application Entrypoint
│   ├── core/
│   │   ├── config.py             # Typed Pydantic Settings
│   │   ├── constants.py          # Enums & Domain Lifecycles
│   │   ├── currency.py           # Currency-Safe Financial Contracts (Money VO)
│   │   ├── exceptions.py         # Standard Exception Hierarchy
│   │   ├── logging.py            # Structured JSON & Contextual Logging
│   │   └── state_machines.py     # State Transition & Permission Validators
│   ├── api/
│   │   ├── schemas.py            # Envelopes, Errors, Pagination Models
│   │   └── v1/
│   │       └── router.py         # Version 1 Master Route Table
│   ├── db/
│   │   ├── base.py               # Declarative Model Base & Custom Mixin
│   │   ├── session.py            # Async Engine & Sessionmaker
│   │   └── models/               # SQLAlchemy Domain Models
│   │       ├── users.py          # Users, Orgs, NGO Profiles
│   │       ├── projects.py       # Projects, Milestones, Geolocation
│   │       ├── financial.py      # Donations, Orders, Authorizations, Captures
│   │       ├── evidence.py       # Evidence Files, Reports, Checks
│   │       └── reviews_impact.py # Reviews, Proof-of-Impact Certificates, Audit
│   └── integrations/             # Vendor-Agnostic Abstraction Interfaces
│       ├── payments/base.py      # Payment Gateway Interface (PayPal)
│       ├── ai/base.py            # Computer Vision & LLM Interface
│       ├── storage/base.py       # Object Storage Interface (S3/MinIO)
│       └── notifications/base.py # Notification Dispatcher Interface
├── alembic/                      # Database Schema Migrations
├── alembic.ini                   # Alembic Migration Configuration
└── pyproject.toml                # Project Build & Dependency Metadata
```

## Running the API Locally
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at:
`http://127.0.0.1:8000/docs`

