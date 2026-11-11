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

## 3. Computer Vision & AI Verification (Phase 6 Completed)
- **Current Status**:
  - `MultiIndicatorAIProvider` integrates YOLOv8n object detection with SHA-256 pinned weights validation (`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`), JPEG Error Level Analysis (ELA) forensics, 64-bit DCT perceptual hashing (pHash), Structural Similarity Index (SSIM) progress comparison, authentic EXIF and GPS proximity verification, safe bounded PDF document inspection, and grounded Explainable AI (XAI) summaries.
  - Asynchronous background execution via durable `VerificationJob` queue with state tracking and retry limits.
  - Cryptographic audit trail appending (`AI_VERIFIED` event chained to `evidence_audit_events`).
  - Streamlit UI and legacy `EnhancedAIVerifier` delegate directly to backend engine.
- **Honest Forensic Limitations & Disclosures**:
  - **Calibration Status**: All reports explicitly output `calibration_status = "UNVALIDATED"`. Confidence scores represent heuristic multi-indicator weighted aggregations, not calibrated empirical Bayes probabilities.
  - **Object Vocabulary**: YOLOv8n is pretrained on the 80 COCO classes. It cannot identify domain-specific medical equipment, educational curricula, or well pump components. Missing classes are reported transparently.
  - **ELA Format Boundaries**: Error Level Analysis is strictly mathematically applicable to lossy JPEG formats. PNG, WebP, and PDF return `NOT_APPLICABLE` without false positives.
  - **Metadata Stripping**: Social messaging apps and mobile web uploads frequently strip EXIF metadata. Absence of GPS coordinates yields `INCONCLUSIVE` geospatial checks rather than failure.
- **Boundaries Preserved**:
  - Advisory findings only: AI verification never triggers automated milestone approval or financial disbursement.
  - Independent reviewer dashboard, dual-key sign-off, and formal dispute resolution are deferred to Phase 7.
  - Escrow fund release and NGO milestone payouts are deferred to Phase 8.
  - Public Proof-of-Impact certificates and NFT generation are deferred to Phase 9.

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


