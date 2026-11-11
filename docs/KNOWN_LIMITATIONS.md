# CharityChain — Known Limitations & Technical Debt

This document maintains an accurate, truthful record of functionality that is currently simulated, incomplete, or requires future phase development.

## 1. Storage & IPFS (Phase 5 Completed)
- **Current Status**:
  - `StorageProviderInterface` implemented with unified synchronous methods.
  - `LocalStorageProvider` handles local disk storage with strict binary magic-byte inspection (PDF, PNG, JPEG), filename sanitization, and canonical path containment (`is_relative_to(base_dir)`).
  - `S3StorageProvider` supports AWS S3 and MinIO with short-lived presigned URLs and custom user metadata.
  - No unsupported claims of public blockchain or IPFS immutability. Provenance is enforced through cryptographic SHA-256 content addressing and deterministic audit hash chains.
- **Boundaries Preserved**:
  - Live AI verification of uploaded evidence is deferred to Phase 6.
  - Independent reviewer approval and human review workflows are deferred to Phase 7.
  - Milestone fund disbursements to NGOs are deferred to Phase 8.

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

## 5. Payments & Financial Transactions (Phase 4 Completed)
- **Current Status**:
  - Full PayPal REST API Orders v2 and Payments v2 integration implemented with `PaymentProviderInterface`.
  - Intent `CAPTURE` implemented with server-side eligibility validation and currency matching.
  - Asymmetric cryptographic webhook signature verification and event deduplication active.
  - Partial and full refund management with remaining refundable balance ceiling enforcement and real-time net funding adjustments.
  - Zero floating-point arithmetic: strictly integer minor units via `Money` value object.
- **Boundaries Preserved**:
  - Captured funds are held in platform accounts; disbursements to NGOs are deferred to Phase 8.
  - Holding captured funds does not constitute a legally chartered banking escrow trust.
  - Milestone evidence upload is deferred to Phase 5.
  - Live AI verification of evidence is deferred to Phase 6.

## 6. Organization Onboarding & Project Management (Phase 3 Completed)
- **Current Status**:
  - Multi-tenant organization onboarding with legal data profiles, verification document uploads, and state machine transitions is fully implemented and tested.
  - Multi-milestone project management with contiguous sequence numbers `1..N`, budget ceiling validation, GPS coordinate bounds, and a live 13-rule pre-submission checklist is fully functional.
  - Independent reviewer queues for organizations and projects feature strict conflict-of-interest prevention.
  - Streamlit transitional interface provides dedicated pages for NGO onboarding, project management, reviewer queue, and public discovery.
- **Boundaries Preserved**:
  - Real PayPal checkout flows and payment authorizations are deferred to Phase 4.
  - General evidence file upload and management (beyond NGO registration documents) are deferred to Phase 5.
  - Live AI verification of project milestone evidence is deferred to Phase 6.
  - Automated escrow fund disbursement upon milestone verification is deferred to Phase 8.
  - Complete Next.js frontend migration is scheduled for Phase 12.


