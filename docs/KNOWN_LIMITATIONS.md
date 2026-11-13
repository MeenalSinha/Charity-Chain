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

## 2. Blockchain & Smart Contracts (Truthful Labeling)
- **Current Status**: No smart contracts are deployed to a public decentralized blockchain. CharityChain does not issue ERC-721/ERC-1155 tokens on Ethereum or Polygon. The platform's verification model uses an **Append-Only Cryptographic Impact Event Ledger** and **Ed25519 Asymmetric Digital Certificates** (RFC 8785 canonical JSON). All marketing and documentation accurately reference "Proof-of-Impact Certificates" and "Cryptographic Event Ledgers" rather than decentralized smart contracts or NFTs.
- **Verification Guarantee**: Tamper-evident authenticity is guaranteed by asymmetric digital signatures and SHA-256 hash chaining, verifiable via standard RFC 7517 JWKS discovery or offline air-gapped CLI tools.

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

## 7. Human Review & Decision Workflows (Phase 7 Completed)
- **Current Status**:
  - Full human review lifecycle implemented with dual-key approval requiring two independent reviewers (Slot 1 and Slot 2).
  - Programmatic conflict-of-interest engine rejects assignments violating organization membership, project creation, case submission, evidence uploading, or cross-slot independence.
  - Workload ceiling strictly enforced at 5 active open tasks per reviewer.
  - Blind review masking protects Reviewer 2 from cognitive anchoring by hiding Reviewer 1's decision/comments until Reviewer 2 submits their independent decision.
  - Multi-round structured revision workflow creates Round N+1 review cases with updated snapshot digests while preserving prior rounds intact.
  - Formal dispute resolution mechanism enables appeals within a configurable window with threaded discussions and independent senior adjudicator assignment.
  - Cryptographic tamper-evident SHA-256 audit ledger with automated mathematical verification (`is_chain_intact`).
- **Boundaries Preserved**:
  - Milestone completion does **NOT** trigger financial disbursements, PayPal payouts, or financial ledger entries. Fund release remains strictly deferred to Phase 8.
  - Public Proof-of-Impact certificates and NFT generation remain strictly deferred to Phase 9.
  - Direct external transactional email dispatch and external webhook callbacks remain strictly deferred to Phase 10.

## 8. Fund Disbursement & Financial Reconciliation (Phase 8 Completed)
- **Current Status**:
  - Full milestone disbursement pipeline implemented with Maker-Checker segregation (Four-Eyes Principle: `requested_by_id != approver_id`).
  - 9-gate automated eligibility verification (`DisbursementEligibilityService`) validates milestone review status, dispute status, escrow balances, and recipient verification.
  - Payout recipient credentials protected at rest with Fernet AES-256 encryption, deterministic SHA-256 blind indexing (`destination_hash`), and masked presentation strings.
  - Modular outbound payout architecture supporting `PayPalPayoutProvider` (PayPal Payouts API v1) and `MockPayoutProvider` for automated testing.
  - Double-entry general ledger with complete Chart of Accounts, strict balance invariants ($\sum \text{Debits} == \sum \text{Credits}$), and append-only reversal mechanics.
  - Three-way automated reconciliation engine classifying discrepancies by standardized taxonomy (`AMOUNT_MISMATCH`, `FEE_MISMATCH`, `STATUS_MISMATCH`, `MISSING_INTERNAL`, `MISSING_EXTERNAL`) with audited resolution workflows.
  - Dedicated Financial Operations management interface integrated into Streamlit UI (`app_ui.py`).
- **Operational & Regulatory Disclosures**:
  - **PayPal Domestic Regulatory Restrictions**: In certain jurisdictions (e.g., PayPal India domestic personal payouts), domestic P2P/payout transfers require specialized merchant regulatory clearance. For local testing and restricted environments, the configurable `MockPayoutProvider` (`PAYOUT_PROVIDER="mock"`) provides high-fidelity simulation.
  - **Banking & Wire Rail Integrations**: `BANK_TRANSFER` recipient methods are fully modeled with AES-256 encrypted routing/account fields and state tracking. Direct clearing via Fedwire/ACH/SEPA requires enterprise banking partner integration.
- **Boundaries Preserved**:
  - Direct external transactional email dispatch and donor notifications remain deferred to Phase 10.
  - Public donor transparency dashboard and aggregated impact heatmaps remain deferred to Phase 11.

## 10. Notifications, Webhooks & Communications (Phase 10 Completed)
- **Current Status**:
  - Full transactional outbox architecture (`NotificationOutbox`, `OutboxService`) guaranteeing zero lost events during database transactions.
  - Canonical domain events (`DomainEvent`) serialized with RFC 8785 canonical JSON and automatic recursive sanitization of credentials, passwords, and payment tokens.
  - In-app notification center with real-time unread badges, paginated inbox listing, and granular user communication preferences protecting mandatory security alerts.
  - Production-grade dual email provider architecture (`SMTPNotificationProvider` and `DevNotificationSink` with fail-closed production protection).
  - Responsive HTML email template engine (`EmailTemplateService`) with XSS escaping and plain-text alternatives.
  - Zero-trust SSRF protection engine (`validate_webhook_url`) blocking private subnets, loopback, cloud metadata (`169.254.169.254`), and dangerous ports with pre-dispatch anti-DNS rebinding checks.
  - Outbound webhook subscriptions with Fernet AES-256 secret encryption at rest, HMAC-SHA256 request signing (`CharityChain-Signature: t=...,v1=...`), timestamp replay window (300s), attempt telemetry, and auto-suspension upon 10 consecutive failures.
  - Background `NotificationWorker` with `run-notification-worker` CLI command.
  - Interactive **Notifications & Webhooks** workspace in Streamlit UI (`app_ui.py`).
- **Operational & Network Disclosures**:
  - **Insecure HTTP Webhooks**: Strictly blocked in production environments (`WEBHOOK_ALLOW_INSECURE_HTTP=False`). Subscribed webhook endpoints must present valid TLS/HTTPS certificates.
  - **Background Worker Runtime**: In development and testing, `NotificationWorker` can run on-demand or with `--once`. In clustered production deployments, run worker processes as dedicated background services.
- **Boundaries Preserved**:
  - Full Next.js frontend migration remains scheduled for Phase 12.

## 11. Analytics, Impact Reporting & Public Transparency (Phase 11 Completed)
- **Current Status**:
  - All legacy analytics methods (relying on unverified TinyDB entries, arbitrary ETH conversions, or unanchored mock counts) have been deprecated and replaced with PostgreSQL database models and live financial records.
  - Multi-dimensional, tenant-isolated NGO analytics (`NGOAnalyticsService`) tracking funding progress, donor retention rates, donation size distributions, AI verification quality diagnostics, and exportable reports.
  - Personalized donor portfolios (`DonorAnalyticsService`) supporting multi-currency segregated giving totals, end-to-end contribution-to-impact lifecycle tracking, and annual tax statements with standard disclaimers.
  - Public transparency portal (`PublicTransparencyService`) exposing verified NGO directories, public project explorers, methodology disclosures, and global platform indicators without exposing PII.
  - Geographic impact mapping (`GeographicAnalyticsService`) with coordinate bounds validation, country density metrics, and automatic coarse centroid obfuscation (~11.1 km radius) for sensitive locations (`is_location_sensitive == True`).
  - Safe data exports (`AnalyticsExportService`) with CWE-1236 CSV formula injection sanitization prepending single quotes to formula trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`).
  - Interactive Streamlit workspaces (`app_ui.py`) for live homepage metrics, donor portfolios, NGO/Platform analytics, Folium geospatial maps, and public transparency.
- **Operational & Analytical Disclosures**:
  - **Tax Disclosure Disclaimer**: Platform annual giving statements are provided strictly for informational purposes based on recorded payments. CharityChain is a technology platform and does not provide formal legal or tax advice.
  - **Location Privacy Granularity**: Sensitive projects coarsen GPS coordinates to 1 decimal place (~11.1 km resolution). Precise coordinates are restricted to authorized project managers and administrators.
- **Boundaries Preserved**:
  - Full Next.js frontend migration remains scheduled for Phase 12.
  - SOC2, Prometheus metrics, and advanced APM remain scheduled for Phase 13.






