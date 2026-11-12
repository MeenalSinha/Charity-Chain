# Proof-of-Impact Event Ledger Architecture

## 1. Executive Summary & Purpose

The **Proof-of-Impact Event Ledger** is CharityChain's immutable, append-only cryptographic event registry. It provides an auditable, verifiable timeline of all milestone achievements, evidence submissions, dual-key human review consensus decisions, disbursements, and impact certificate lifecycle events.

The Impact Ledger guarantees:
- **Append-Only Immutability**: Historical events cannot be overwritten, modified, or silently excised without mathematically breaking the cryptographic hash chain.
- **Strict Monotonicity**: Sequences progress strictly as $1, 2, 3, \dots, n$ without gaps, reordering, or duplicate sequence indices.
- **Cryptographic Hash Chaining**: Every event links to its immediate predecessor via a SHA-256 domain-separated digest.
- **Signed Periodic Checkpoints**: High-water mark checkpoints signed with platform Ed25519 issuer keys anchor the event stream against rollback attacks.
- **Independent Auditability**: Any third party, auditor, or public verifier can mathematically verify stream continuity and tamper resistance without privileged platform credentials.

---

## 2. Separation of Ledgers: Financial vs. Impact

A fundamental tenet of CharityChain's Phase 8 & 9 architecture is the strict separation between the **Double-Entry Financial General Ledger** and the **Proof-of-Impact Event Ledger**:

```
+--------------------------------------------------------------------------------------------------+
|                                    CHARITYCHAIN DUAL-LEDGER TOPOLOGY                            |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +---------------------------------------------+   +------------------------------------------+ |
|   |         FINANCIAL GENERAL LEDGER            |   |       PROOF-OF-IMPACT EVENT LEDGER       | |
|   |                 (Phase 8)                   |   |                 (Phase 9)                | |
|   +---------------------------------------------+   +------------------------------------------+ |
|   | * Double-entry accounting principles        |   | * Cryptographic event-sourcing stream    | |
|   | * Debits and credits balance to ZERO        |   | * Append-only monotonic sequence (1..n)   | |
|   | * Multi-currency minor unit arithmetic      |   | * RFC 8785 Canonical JSON payloads       | |
|   | * Tracks: donations, allocations, payouts,  |   | * Tracks: evidence, reviews, approvals,  | |
|   |   processing fees, and completed refunds    |   |   disbursements, certificate issuance    | |
|   | * Enforces Maker-Checker financial controls |   | * Domain-separated SHA-256 hash chaining | |
|   | * Target audience: CPAs, auditors, banks    |   | * Target audience: Donors, public, NGOs  | |
|   +---------------------------------------------+   +------------------------------------------+ |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

### Key Architectural Boundaries
1. **Financial Ledger (`LedgerEntry`, `LedgerAccount`)**:
   - Manages money, balances, internal milestone escrow reservations, and outbound payout disbursements.
   - Enforces the universal accounting invariant: $\sum \text{Debits} - \sum \text{Credits} = 0$.
   - Never contains arbitrary unstructured milestone claims or evidence blobs.

2. **Impact Ledger (`ImpactLedgerStream`, `ImpactLedgerEvent`, `ImpactLedgerCheckpoint`)**:
   - Manages programmatic attestations of charitable execution and verification proof.
   - Preserves an unalterable history of governance actions and cryptographic certificates.
   - Never directly moves currency balances or triggers banking payouts.

---

## 3. Event Stream Aggregate Design

The Impact Ledger is structured around isolated aggregate streams:
- **Milestone Aggregate (`milestone-{milestone_id}`)**: Primary stream recording all activities for a specific charitable milestone from evidence upload to certificate revocation.
- **Project Aggregate (`project-{project_id}`)**: Project-level governance and lifecycle events.
- **Global Stream (`global-stream`)**: Platform-wide security and key rotation events.

### Stream Head Tracking
The `ImpactLedgerStream` entity maintains:
- `stream_id`: Unique deterministic identifier (e.g., `milestone-3fa85f64-5717-4562-b3fc-2c963f66afa6`).
- `aggregate_type`: `"MILESTONE"`, `"PROJECT"`, or `"GLOBAL"`.
- `aggregate_id`: Foreign key reference to the underlying aggregate entity.
- `current_sequence_number`: Latest monotonically assigned event sequence (starts at 0).
- `head_event_hash`: SHA-256 hash of the most recent event (initialized to `GENESIS_HASH = "0" * 64`).
- `status`: `"ACTIVE"` or `"SEALED"`.

---

## 4. Cryptographic Hash-Chaining Specification

Every ledger event is bound to the historical chain through a deterministic, domain-separated cryptographic formula.

### Formula Specification

$$\text{current\_event\_hash} = \text{SHA-256}(\text{previous\_event\_hash} \mid \text{stream\_id} \mid \text{sequence\_number} \mid \text{event\_type} \mid \text{payload\_hash})$$

Where:
- $\text{previous\_event\_hash}$: 64-character hex SHA-256 digest of event $n-1$ (or `GENESIS_HASH` for sequence 1).
- $\text{stream\_id}$: Target stream identifier.
- $\text{sequence\_number}$: Monotonically increasing 1-based integer ($n$).
- $\text{event\_type}$: Standardized `ImpactLedgerEventType` string.
- $\text{payload\_hash}$: SHA-256 hex digest of the RFC 8785 Canonical JSON payload string.
- $\mid$: ASCII pipe separator (`0x7C`).

```
+-------------------------------------------------------------------------------------------------+
|                                     HASH-CHAIN CONTINUITY FLOW                                  |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   +-----------------------+       +-----------------------+       +-----------------------+     |
|   |     Event Seq #1      |       |     Event Seq #2      |       |     Event Seq #3      |     |
|   |-----------------------|       |-----------------------|       |-----------------------|     |
|   | Prev: 00000000...0000 |       | Prev: Hash(Event #1)  |       | Prev: Hash(Event #2)  |     |
|   | Type: EVIDENCE_SUBMIT |       | Type: DUAL_KEY_APPRVD |       | Type: CERT_ISSUED     |     |
|   | Payload: Canonical    |       | Payload: Canonical    |       | Payload: Canonical    |     |
|   | Hash: Hash(Event #1)  | ----> | Hash: Hash(Event #2)  | ----> | Hash: Hash(Event #3)  |     |
|   +-----------------------+       +-----------------------+       +-----------------------+     |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 5. Signed Cryptographic Checkpoints

To ensure stream integrity cannot be rewritten from genesis in a compromised database environment, the platform creates signed checkpoints:
- **Frequency**: Automated every 50 events (`IMPACT_LEDGER_AUTO_CHECKPOINT_INTERVAL`) or triggered on-demand via administrative REST API / CLI.
- **Content**: Canonical checkpoint payload containing `stream_id`, `sequence_number`, `head_hash`, and timestamp.
- **Signature**: Signed by the active Ed25519 platform issuer key.
- **Audit Verification**: Auditors compare stored stream heads against publicly published or externally archived checkpoint signatures.

---

## 6. Tamper Detection & Independent Verification Algorithm

The verification algorithm (`ImpactLedgerService.verify_stream_chain`) performs 4 mathematical checks:

```python
def verify_stream_chain(db: Session, stream_id: str) -> Dict[str, Any]:
    # 1. Monotonic sequence check: seq == index (1, 2, 3...)
    # 2. Previous hash link check: ev.previous_event_hash == expected_prev_hash
    # 3. Payload integrity check: SHA-256(ev.payload_json) == ev.payload_hash
    # 4. Current event recomputation: compute_event_hash(...) == ev.current_event_hash
```

If any database row is altered, deleted, or injected:
- Mutated payload $\implies$ `Payload tampering detected at sequence X`.
- Altered hash pointer $\implies$ `Hash chain broken at sequence X`.
- Sequence gap or injection $\implies$ `Sequence break at index X`.
- Altered head $\implies$ `Stream head hash mismatch`.

---

## 7. Supported Impact Ledger Event Types

The ledger records formal lifecycle events defined in `ImpactLedgerEventType`:
1. `ORGANIZATION_VERIFIED`: NGO passes formal onboarding verification.
2. `PROJECT_APPROVED`: Project approved by platform administrators.
3. `PROJECT_ACTIVATED`: Project opened for public donations.
4. `DONATION_CAPTURED`: Funds captured via PayPal payment gateway.
5. `EVIDENCE_SUBMITTED`: Field evidence submitted for milestone verification.
6. `EVIDENCE_VERSION_CREATED`: Revision or updated evidence version uploaded.
7. `EVIDENCE_INTEGRITY_VERIFIED`: SHA-256 hash and metadata validated against storage.
8. `AI_VERIFICATION_COMPLETED`: YOLOv8, ELA, EXIF, and LLM analysis findings generated.
9. `MILESTONE_REVIEW_SUBMITTED`: Evidence submitted for formal human review.
10. `FIRST_REVIEWER_DECISION`: Primary reviewer votes on milestone evidence.
11. `SECOND_REVIEWER_DECISION`: Secondary reviewer votes on milestone evidence.
12. `DUAL_KEY_APPROVAL_FINALIZED`: Dual-key consensus achieved approving milestone.
13. `DISPUTE_RESOLVED`: Governance dispute adjudicated by platform administrators.
14. `MILESTONE_COMPLETED`: Milestone confirmed complete.
15. `DISBURSEMENT_CONFIRMED`: Financial payout execution confirmed to NGO account.
16. `FINANCIAL_RECONCILIATION_COMPLETED`: Milestone accounts balanced to zero variance.
17. `CERTIFICATE_ISSUED`: Cryptographic Proof-of-Impact certificate issued.
18. `CERTIFICATE_SUSPENDED`: Certificate temporarily suspended during investigation.
19. `CERTIFICATE_REVOKED`: Certificate permanently revoked under governance policy.
20. `CERTIFICATE_SUPERSEDED`: Certificate superseded by newer revised issuance.

