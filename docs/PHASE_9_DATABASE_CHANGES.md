# CharityChain — Phase 9 Database Changes & Migration Reference
## Schema Enhancements, Migration 009, Tables, Constraints, and Indices

---

## 1. Migration Overview

- **Migration Identifier**: `009_phase9_impact_ledger_and_certificates.py`
- **Revision ID**: `009`
- **Revises**: `008` (Double-entry financial ledger and payouts)
- **Target Subsystems**: Proof-of-Impact Certificates, Asymmetric Signing Keys, Append-Only Impact Event Ledger, and Ledger Checkpoints.

---

## 2. Table Modifications

### 2.1 Table: `impact_certificates` (Altered / Modernized)

The legacy placeholder table `impact_certificates` was augmented with complete cryptographic, governance, and verification metadata.

| Column Name | Data Type | Nullable | Constraints & Defaults | Description |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary Key | Canonical certificate identifier |
| `certificate_number` | `String(64)` | No | Unique, Indexed | Human-readable identifier (e.g., `CC-CERT-2026-000142`) |
| `verification_id` | `String(64)` | No | Unique, Indexed | High-entropy public verification token (QR slug) |
| `certificate_type` | `Enum(CertificateType)` | No | Indexed | `CONTRIBUTION` or `MILESTONE_COMPLETION` |
| `status` | `Enum(CertificateStatus)` | No | Indexed, Default: `ISSUED` | `DRAFT`, `PENDING_ISSUANCE`, `ISSUED`, `SUSPENDED`, `REVOKED`, `SUPERSEDED` |
| `recipient_name` | `String(255)` | Yes | | Display name of recipient or organization |
| `donor_id` | `UUID` / `String(36)` | Yes | FK -> `users.id`, Indexed | Associated donor user ID (if contribution) |
| `project_id` | `UUID` / `String(36)` | No | FK -> `projects.id`, Indexed | Associated charitable project ID |
| `milestone_id` | `UUID` / `String(36)` | No | FK -> `milestones.id`, Indexed | Associated milestone ID |
| `contribution_id` | `UUID` / `String(36)` | Yes | FK -> `donations.id`, Indexed | Associated donation record |
| `public_attribution` | `Boolean` | No | Default: `False` | Donor consent flag for public display |
| `claims_data` | `JSON` | No | Default: `{}` | Complete structured claim payload (RFC 8785) |
| `claims_hash` | `String(64)` | No | Indexed | SHA-256 of canonical JSON claims |
| `canonical_claims_b64`| `Text` | Yes | | Base64-encoded canonical JSON string |
| `signature` | `String(255)` | No | | Base64-encoded Ed25519 digital signature |
| `signature_algorithm`| `String(32)` | No | Default: `Ed25519` | Asymmetric signature algorithm |
| `signature_key_id` | `String(64)` | No | Indexed | Issuer key identifier (`kid`) |
| `pdf_storage_path` | `String(512)` | Yes | | Object storage or local path to generated PDF |
| `pdf_sha256` | `String(64)` | Yes | | SHA-256 integrity hash of rendered PDF file |
| `supersedes_id` | `UUID` / `String(36)` | Yes | FK -> `impact_certificates.id` | Previous certificate superseded by this version |
| `superseded_by_id` | `UUID` / `String(36)` | Yes | FK -> `impact_certificates.id` | Successor certificate superseding this version |
| `revocation_reason` | `String(255)` | Yes | | Explanatory note if revoked or suspended |
| `revocation_code` | `Enum(...)` | Yes | | Structured revocation reason code |
| `revoked_at` | `DateTime(timezone=True)` | Yes | | Timestamp when revocation took effect |
| `revoked_by_user_id`| `UUID` / `String(36)` | Yes | FK -> `users.id` | Operator or governance authority executing revocation |
| `issued_at` | `DateTime(timezone=True)` | No | Indexed, Default: UTC Now | Official issuance timestamp |
| `created_at` | `DateTime(timezone=True)` | No | Default: UTC Now | Row creation timestamp |
| `updated_at` | `DateTime(timezone=True)` | No | Default: UTC Now | Row update timestamp |

---

## 3. New Tables Created

### 3.1 Table: `certificate_status_events`
Tracks append-only lifecycle transitions (suspension, reinstatement, revocation, supersession).

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary key |
| `certificate_id` | `UUID` / `String(36)` | No | FK -> `impact_certificates.id` (ON DELETE CASCADE) |
| `previous_status` | `Enum(CertificateStatus)` | No | Prior lifecycle state |
| `new_status` | `Enum(CertificateStatus)` | No | New lifecycle state |
| `reason` | `Text` | Yes | Human-readable explanation |
| `reason_code` | `String(64)` | Yes | Machine-readable status transition code |
| `actor_user_id` | `UUID` / `String(36)` | Yes | FK -> `users.id` (actor triggering change) |
| `created_at` | `DateTime(timezone=True)` | No | UTC timestamp of event |

---

### 3.2 Table: `issuer_signing_keys`
Registers asymmetric Ed25519 signing keys and lifecycle states.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary key |
| `key_id` | `String(64)` | No | Unique Key ID (`kid`) in JWKS |
| `algorithm` | `String(32)` | No | Default: `Ed25519` |
| `public_key_b64` | `String(128)` | No | Base64-encoded raw 32-byte public key |
| `private_key_b64` | `Text` | Yes | Base64-encoded private key (local dev only) |
| `is_active` | `Boolean` | No | Default: `False`, Exactly one active key |
| `status` | `Enum(KeyStatus)` | No | `PROVISIONED`, `ACTIVE`, `RETIRED`, `COMPROMISED` |
| `created_at` | `DateTime(timezone=True)` | No | Generation timestamp |
| `activated_at` | `DateTime(timezone=True)` | Yes | Activation timestamp |
| `retired_at` | `DateTime(timezone=True)` | Yes | Retirement timestamp |
| `compromised_at` | `DateTime(timezone=True)` | Yes | Compromise declaration timestamp |
| `notes` | `Text` | Yes | Administrative notes / incident reports |

---

### 3.3 Table: `impact_ledger_streams`
Partitions the append-only impact event log by entity scope.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary key |
| `stream_type` | `Enum(ImpactLedgerStreamType)`| No | `MILESTONE`, `PROJECT`, `ORGANIZATION`, `GLOBAL` |
| `entity_id` | `UUID` / `String(36)` | No | Referenced entity ID |
| `current_sequence` | `BigInteger` | No | Default: `0`, Monotonically increasing sequence number |
| `latest_event_hash`| `String(64)` | Yes | Hash of latest chained event |
| `created_at` | `DateTime(timezone=True)` | No | Stream initialization timestamp |
| `updated_at` | `DateTime(timezone=True)` | No | Last event appended timestamp |

**Indexes & Constraints**:
- `uq_impact_ledger_streams_type_entity`: Unique on `(stream_type, entity_id)`.

---

### 3.4 Table: `impact_ledger_events`
The append-only cryptographic event chain.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary key |
| `stream_id` | `UUID` / `String(36)` | No | FK -> `impact_ledger_streams.id` |
| `sequence_number` | `BigInteger` | No | Monotonic 1-based index within stream |
| `event_type` | `Enum(ImpactLedgerEventType)` | No | Lifecycle event classification |
| `event_timestamp` | `DateTime(timezone=True)` | No | UTC event occurrence time |
| `source_event_id` | `String(64)` | Yes | Foreign event identifier (for deduplication) |
| `source_event_type`| `String(64)` | Yes | Domain source entity type |
| `actor_user_id` | `UUID` / `String(36)` | Yes | FK -> `users.id` |
| `actor_role` | `String(64)` | Yes | Role of actor at time of event |
| `payload_data` | `JSON` | No | Event data payload (canonicalized RFC 8785) |
| `payload_hash` | `String(64)` | No | SHA-256 of canonical payload |
| `previous_event_hash`| `String(64)` | No | SHA-256 hash of event $i-1$ (Genesis: 64 zeros) |
| `event_hash` | `String(64)` | No | SHA-256 hash: $(prev\_hash \parallel seq \parallel type \parallel payload\_hash)$ |
| `signature` | `String(255)` | Yes | Optional direct event signature |
| `created_at` | `DateTime(timezone=True)` | No | Record insertion timestamp |

**Indexes & Constraints**:
- `uq_impact_ledger_events_stream_seq`: Unique on `(stream_id, sequence_number)`
- `uq_impact_ledger_events_source_dedup`: Unique on `(stream_id, source_event_type, source_event_id)`

---

### 3.5 Table: `impact_ledger_checkpoints`
Signed state checkpoints summarizing stream history.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` / `String(36)` | No | Primary key |
| `stream_id` | `UUID` / `String(36)` | No | FK -> `impact_ledger_streams.id` |
| `start_sequence` | `BigInteger` | No | First sequence covered by checkpoint |
| `end_sequence` | `BigInteger` | No | Last sequence covered by checkpoint |
| `total_events` | `Integer` | No | Count of events in checkpoint block |
| `state_root_hash` | `String(64)` | No | Merkle root or chained cumulative hash |
| `signature` | `String(255)` | No | Base64 Ed25519 signature of state root |
| `key_id` | `String(64)` | No | Issuer key ID used for signature |
| `created_at` | `DateTime(timezone=True)` | No | Checkpoint creation timestamp |

---

## 4. Migration Execution & Verification

Run the migration using Alembic:
```bash
alembic upgrade head
```

Verify table status:
```bash
alembic current
```
Output:
```
009 (head)
```

