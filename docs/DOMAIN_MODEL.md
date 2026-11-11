# CharityChain — Domain Model Specification
**Phase 1 Deliverable — Bounded Contexts, Entity Definitions & Invariant Rules**

---

## 1. Domain Modeling Principles

CharityChain's domain model represents a mission-critical, auditable charitable platform where funds are tied to verified real-world milestones. The domain model strictly enforces:
1. **Value Objects for Financial Integrity**: All monetary values are expressed as `Money(minor_units, currency)` using integer minor units (e.g. cents). Floating-point data types are strictly prohibited in database columns and calculations.
2. **Deterministic State Machines**: Lifecycles for Projects, Milestones, Payment Orders, Captures, Disbursements, and Evidence are governed by formal transition rules with explicit actor authorization.
3. **Audit Trail Completeness**: All entities inherit immutable audit attributes (`id`, `created_at`, `updated_at`, `is_active`) and emit structured audit events upon critical mutations.
4. **Separation of Authorization vs Capture vs Disbursement**: Donations do not immediately disburse to NGOs. Funds are authorized, held, captured upon milestone completion, and disbursed via verified settlement.

---

## 2. Entity Specifications by Bounded Context

### 2.1 Identity & Access Management (IAM)

#### `User`
Primary principal identity representing donors, NGO officers, and platform reviewers.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `email`: String(255), unique, indexed, case-insensitive
  - `hashed_password`: String(255), Argon2id / bcrypt hash
  - `full_name`: String(255), nullable
  - `role`: Enum (`DONOR`, `NGO_ADMIN`, `REVIEWER`, `SUPER_ADMIN`), default: `DONOR`
  - `is_verified`: Boolean, default: `False`
  - `is_active`: Boolean, default: `True`
  - `created_at`, `updated_at`: UTC DateTime
- **Relationships**:
  - One-to-One with `UserProfile`
  - One-to-Many with `OrganizationMembership`
  - One-to-Many with `Donation`
  - One-to-Many with `ReviewDecisionRecord`

#### `Organization` & `NGOProfile`
Represents legally registered non-profit and charitable entities.
- **Attributes (`Organization`)**:
  - `id`: UUID (Primary Key)
  - `name`: String(255), unique
  - `legal_registration_number`: String(100), unique, nullable
  - `tax_id`: String(100), nullable
  - `country`: String(2), ISO 3166-1 alpha-2
  - `status`: Enum (`PENDING_APPROVAL`, `ACTIVE`, `SUSPENDED`, `REJECTED`)
  - `verification_level`: Enum (`TIER_0_UNVERIFIED`, `TIER_1_BASIC`, `TIER_2_AUDITED`)
- **Attributes (`NGOProfile`)**:
  - `id`: UUID (Primary Key)
  - `organization_id`: UUID (Foreign Key -> `Organization.id`)
  - `mission_statement`: Text
  - `website_url`: String(255), nullable
  - `paypal_payout_email`: String(255), nullable
  - `verified_at`: UTC DateTime, nullable

---

### 2.2 Projects & Milestones

#### `Project`
Charitable initiative with defined target funding, timeframe, and discrete milestones.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `organization_id`: UUID (Foreign Key -> `Organization.id`)
  - `category_id`: UUID (Foreign Key -> `ProjectCategory.id`)
  - `title`: String(255), indexed
  - `slug`: String(255), unique, indexed
  - `description`: Text
  - `target_amount_minor`: BigInteger, nullable=False (Goal in cents)
  - `currency`: String(3), ISO 4217 code (e.g. 'USD')
  - `status`: Enum (`DRAFT`, `SUBMITTED`, `UNDER_REVIEW`, `APPROVED`, `ACTIVE`, `FUNDED`, `COMPLETED`, `SUSPENDED`, `CANCELLED`)
  - `beneficiary_count`: Integer, default: 0
  - `start_date`, `end_date`: Date
- **Invariants**:
  - Target amount must be positive (> 0).
  - Sum of milestone amounts must not exceed project target amount.
  - Cannot transition to `ACTIVE` without at least one defined milestone and approved organization status.

#### `Milestone`
Discrete phase of project execution required for fund release.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `project_id`: UUID (Foreign Key -> `Project.id`)
  - `sequence_order`: Integer, indexed (1, 2, 3...)
  - `title`: String(255)
  - `description`: Text
  - `target_amount_minor`: BigInteger, nullable=False
  - `currency`: String(3)
  - `status`: Enum (`NOT_STARTED`, `IN_PROGRESS`, `EVIDENCE_SUBMITTED`, `IN_REVIEW`, `VERIFIED`, `DISBURSED`, `FAILED`, `CANCELLED`)
  - `due_date`: Date, nullable
  - `verified_at`: UTC DateTime, nullable
- **Invariants**:
  - Milestones must be completed in sequential order unless configured as concurrent.
  - Funds cannot be disbursed until milestone status is `VERIFIED`.

#### `ProjectLocation`
Geographic coordinates defining the intended site of project activity.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `project_id`: UUID (Foreign Key -> `Project.id`)
  - `latitude`: Float (-90.0 to 90.0)
  - `longitude`: Float (-180.0 to 180.0)
  - `radius_meters`: Float, default: 1000.0 (Acceptable GPS radius for evidence)
  - `address_text`: String(500), nullable

---

### 2.3 Financials & Payments

#### `Donation`
Donor financial contribution intent and attribution.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `user_id`: UUID (Foreign Key -> `User.id`, nullable for guest donations)
  - `project_id`: UUID (Foreign Key -> `Project.id`)
  - `amount_minor`: BigInteger (Contribution amount in cents)
  - `currency`: String(3)
  - `status`: Enum (`PENDING`, `AUTHORIZED`, `CAPTURED`, `REFUNDED`, `FAILED`)
  - `is_anonymous`: Boolean, default: `False`
  - `donor_name`: String(255), nullable

#### `PaymentOrder`
Gateway-level transaction tracking (e.g. PayPal Order).
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `donation_id`: UUID (Foreign Key -> `Donation.id`)
  - `gateway`: String(50), e.g. "PAYPAL"
  - `gateway_order_id`: String(100), unique, indexed
  - `intent`: Enum (`AUTHORIZE`, `CAPTURE`), default: `AUTHORIZE`
  - `amount_minor`: BigInteger
  - `currency`: String(3)
  - `status`: Enum (`CREATED`, `SAVED`, `APPROVED`, `VOIDED`, `COMPLETED`, `PAYER_ACTION_REQUIRED`)
  - `expires_at`: UTC DateTime

#### `PaymentAuthorization` & `PaymentCapture`
Separated records of fund authorization (hold) and capture (settlement).
- **Attributes (`PaymentAuthorization`)**:
  - `id`: UUID (Primary Key)
  - `order_id`: UUID (Foreign Key -> `PaymentOrder.id`)
  - `gateway_auth_id`: String(100), unique, indexed
  - `amount_minor`: BigInteger
  - `status`: Enum (`CREATED`, `CAPTURED`, `DENIED`, `EXPIRED`, `VOIDED`, `PENDING`)
  - `expiration_date`: UTC DateTime
- **Attributes (`PaymentCapture`)**:
  - `id`: UUID (Primary Key)
  - `authorization_id`: UUID (Foreign Key -> `PaymentAuthorization.id`)
  - `gateway_capture_id`: String(100), unique, indexed
  - `amount_minor`: BigInteger
  - `net_amount_minor`: BigInteger, nullable (After gateway fees)
  - `fee_amount_minor`: BigInteger, nullable
  - `status`: Enum (`COMPLETED`, `DECLINED`, `PARTIALLY_REFUNDED`, `PENDING`, `REFUNDED`)

#### `Disbursement`
Actual transfer of verified funds to the NGO's account.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `milestone_id`: UUID (Foreign Key -> `Milestone.id`)
  - `organization_id`: UUID (Foreign Key -> `Organization.id`)
  - `amount_minor`: BigInteger
  - `currency`: String(3)
  - `gateway_payout_batch_id`: String(100), nullable
  - `status`: Enum (`PENDING_APPROVAL`, `QUEUED`, `PROCESSING`, `SUCCESS`, `FAILED`)
  - `executed_at`: UTC DateTime, nullable

---

### 2.4 Evidence & Verification

#### `EvidenceFile`
Digital file submitted by NGO proving milestone completion.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `milestone_id`: UUID (Foreign Key -> `Milestone.id`)
  - `uploader_id`: UUID (Foreign Key -> `User.id`)
  - `file_name`: String(255)
  - `mime_type`: String(100) (Strictly validated against allowed list)
  - `file_size_bytes`: BigInteger (Max 10 MB)
  - `storage_path`: String(500)
  - `sha256_hash`: String(64), unique, indexed (Cryptographic content digest)
  - `phash`: String(64), nullable (Perceptual hash for image deduplication)
  - `exif_metadata`: JSON, nullable
  - `latitude`, `longitude`: Float, nullable (Extracted from EXIF GPS)

#### `VerificationReport` & `VerificationCheckRecord`
Forensic analysis output aggregated across automated AI checks.
- **Attributes (`VerificationReport`)**:
  - `id`: UUID (Primary Key)
  - `evidence_file_id`: UUID (Foreign Key -> `EvidenceFile.id`)
  - `overall_confidence`: Float (0.0 to 1.0, normalized only across executed checks)
  - `status`: Enum (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`)
  - `is_tampered_indicator`: Boolean, default: `False`
  - `summary`: Text
- **Attributes (`VerificationCheckRecord`)**:
  - `id`: UUID (Primary Key)
  - `report_id`: UUID (Foreign Key -> `VerificationReport.id`)
  - `check_type`: Enum (`YOLO_OBJECT_DETECTION`, `ERROR_LEVEL_ANALYSIS`, `PHASH_DUPLICATE_CHECK`, `SSIM_COMPARISON`, `GPS_RADIUS_CHECK`)
  - `status`: Enum (`PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`, `PENDING`)
  - `score`: Float, nullable
  - `details`: JSON

---

### 2.5 Reviews, Impact & Audit

#### `ReviewTask` & `ReviewDecisionRecord`
Formal human governance records verifying or rejecting evidence.
- **Attributes (`ReviewTask`)**:
  - `id`: UUID (Primary Key)
  - `milestone_id`: UUID (Foreign Key -> `Milestone.id`)
  - `report_id`: UUID (Foreign Key -> `VerificationReport.id`)
  - `status`: Enum (`PENDING_REVIEW`, `IN_REVIEW`, `DECIDED`, `CANCELLED`)
  - `assigned_reviewer_id`: UUID (Foreign Key -> `User.id`, nullable)
- **Attributes (`ReviewDecisionRecord`)**:
  - `id`: UUID (Primary Key)
  - `task_id`: UUID (Foreign Key -> `ReviewTask.id`)
  - `reviewer_id`: UUID (Foreign Key -> `User.id`)
  - `decision`: Enum (`APPROVED`, `REJECTED`, `MORE_INFO_REQUESTED`)
  - `notes`: Text
  - `decided_at`: UTC DateTime

#### `ImpactCertificate`
Verifiable digital record of verified milestone impact awarded to donors.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `donation_id`: UUID (Foreign Key -> `Donation.id`)
  - `milestone_id`: UUID (Foreign Key -> `Milestone.id`)
  - `certificate_number`: String(100), unique, indexed
  - `sha256_digest`: String(64) (Cryptographic verification hash)
  - `token_units`: Integer (Calculated impact points, e.g. 100 per $1)
  - `issued_at`: UTC DateTime

#### `AuditEvent`
Immutable append-only ledger of platform mutations.
- **Attributes**:
  - `id`: UUID (Primary Key)
  - `entity_name`: String(100), indexed
  - `entity_id`: UUID, indexed
  - `action`: String(50), e.g. "STATUS_CHANGE", "DISBURSEMENT_INITIATED"
  - `actor_id`: UUID, nullable
  - `before_state`: JSON, nullable
  - `after_state`: JSON, nullable
  - `correlation_id`: String(100), indexed
  - `created_at`: UTC DateTime

