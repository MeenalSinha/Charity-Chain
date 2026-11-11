# CharityChain — Known Limitations & Technical Debt

This document maintains an accurate, truthful record of functionality that is currently simulated, incomplete, or requires future phase development.

## 1. Storage & IPFS
- **Current Status**: Real Web3.Storage/Pinata requires active API tokens. When in default/demo mode or when tokens are unconfigured, local SHA-256 content hashes are generated and stored locally (`data/evidence/`).
- **Target (Phase 5)**: S3-compatible object storage (MinIO/AWS S3) with cryptographic content-addressing and metadata persistence.

## 2. Blockchain & Smart Contracts
- **Current Status**: No smart contracts are deployed to an active blockchain. Ethereum/Polygon mentions in the original prototype were simulated using local Python logic and TinyDB. The current release explicitly labels these as "Milestone-Locked Funds" and "Proof-of-Impact Digital Certificates" rather than true on-chain transactions or NFTs.
- **Target (Phase 4 & 9)**: Real payment gateway escrow/authorization holds (PayPal REST API) and cryptographically verifiable digital certificates.

## 3. Computer Vision & AI Verification
- **Current Status**:
  - YOLOv8 (`yolov8n.pt`) is loaded locally for genuine object detection. If weights are missing, the check is marked `UNAVAILABLE`, not fabricated.
  - Error Level Analysis (ELA) and perceptual hash (pHash) are implemented as image forensics indicators, but cannot definitively prove forgery alone.
  - EXIF GPS parsing extracts latitude and longitude when present. Modern smartphones often strip EXIF GPS during upload or social sharing. When missing, the status is marked `UNAVAILABLE` rather than defaulting to hardcoded distances.
  - No live LLM (OpenAI/Gemini) is connected by default. Unconfigured queries return clear rule-based system status notes.
- **Target (Phase 6)**: Multi-model ensemble with cloud-hosted inference, robust EXIF forensics, reverse-image lookup, and LLM multimodal verification.

## 4. Database & Concurrency
- **Current Status**: TinyDB stores data in a single JSON file (`data/charity.db`). While stabilized for local single-process development, it does not support ACID transactions, high-concurrency writes, or complex relational queries.
- **Target (Phase 1 & Database Migration)**: PostgreSQL with SQLAlchemy ORM and Alembic migrations.

## 5. Payments & Financial Transactions
- **Current Status**: Original prototype used simulated ETH numbers (e.g. `2.5 * 10^18`).
- **Target (Phase 1 & Phase 4)**: Currency-safe data types (minor units in integer or Decimal), distinct lifecycle states for authorization, capture, void, refund, and payout via PayPal REST API.

