# Proof-of-Impact Certificates Specification

## 1. Overview & Architectural Philosophy

CharityChain Digital Impact Certificates are authentic, cryptographically signed attestations that a charitable contribution was genuinely captured and applied to a verified, completed project milestone.

Unlike cosmetic badges or unsupported blockchain NFT assertions, CharityChain certificates are:
- **Rooted in Real Evidence**: Issued only after physical evidence (photos, documents, receipts) has been subjected to AI forensic analysis and dual-key human review consensus.
- **Bound to Financial Settlements**: Tied directly to settled payment gateway captures and verified double-entry ledger accounts.
- **Cryptographically Non-Repudiable**: Digitally signed using Edwards-curve Digital Signature Algorithm (Ed25519) over RFC 8785 canonical JSON claims.
- **Publicly Verifiable**: Independently verifiable via mobile QR code scans and offline cryptographic proof envelopes.

---

## 2. Certificate Typology

| Certificate Type | Target Entity | Core Purpose | Required Eligibility Conditions |
| :--- | :--- | :--- | :--- |
| `CONTRIBUTION_IMPACT` | Individual Donor | Proves donor's funds supported an approved, completed milestone | Confirmed payment capture, verified NGO, dual-key review approval, clean dispute status |
| `DISBURSEMENT_ATTESTATION` | NGO / Partner | Proves milestone funds were disbursed and reconciled to zero balance | Maker-checker approved payout execution, bank settlement confirmation, financial ledger balancing |

---

## 3. The 9-Gate Certificate Eligibility Engine

The `CertificateEligibilityService` evaluates 9 strict gates before any certificate is signed or issued. Simple milestone status strings (e.g., `status = "COMPLETED"`) are **never** accepted on their own.

```
+-------------------------------------------------------------------------------------------------+
|                                 9-GATE CERTIFICATE ELIGIBILITY PIPELINE                          |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [ 1. Donation Exists ]                                                                        |
|            |                                                                                    |
|            v                                                                                    |
|   [ 2. Confirmed Payment Capture ] ------> Must have PaymentCapture(status="COMPLETED")         |
|            |                                                                                    |
|            v                                                                                    |
|   [ 3. Net Positive Balance ] -----------> Total Captures - Completed Refunds > 0               |
|            |                                                                                    |
|            v                                                                                    |
|   [ 4. Milestone & Project Active ] -----> Milestone exists and links to valid Project          |
|            |                                                                                    |
|            v                                                                                    |
|   [ 5. Organization Verified ] ----------> Organization.status == OrganizationStatus.VERIFIED   |
|            |                                                                                    |
|            v                                                                                    |
|   [ 6. Dual-Key Human Approval ] --------> ReviewCase(final_decision="APPROVE", status="APPROVED")|
|            |                                                                                    |
|            v                                                                                    |
|   [ 7. No Unresolved Disputes ] ---------> No open ReviewDispute records for this milestone     |
|            |                                                                                    |
|            v                                                                                    |
|   [ 8. Idempotency Check ] --------------> No existing ACTIVE or ISSUED certificate for donation|
|            |                                                                                    |
|            v                                                                                    |
|   [ 9. Optional Payout Gate ] -----------> For DISBURSEMENT_ATTESTATION, payout must be settled |
|            |                                                                                    |
|            v                                                                                    |
|   [ ELIGIBILITY APPROVED: Proceed to Cryptographic Ed25519 Signing Pipeline ]                  |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 4. Canonical Claims Schema

The signed claims payload is structured strictly according to the following canonical format:

```json
{
  "schema_version": "v9.0.0",
  "certificate_type": "CONTRIBUTION_IMPACT",
  "certificate_number": "CC-CERT-20261112-9A4B3C2D",
  "verification_id": "v_id_38f2a1b9c8d7e6f543210fedcba98765",
  "issuer": {
    "id": "charitychain-core-v1",
    "name": "CharityChain Platform",
    "public_url": "https://api.charitychain.org"
  },
  "issued_at": "2026-11-12T12:00:00Z",
  "donor": {
    "display_name": "Jane Doe",
    "is_anonymous": false
  },
  "contribution": {
    "donation_id": "d-7890-abcd-1234",
    "amount_cents": 5000,
    "currency": "USD"
  },
  "project": {
    "id": "p-1111-2222-3333",
    "title": "Clean Water for Turkana"
  },
  "milestone": {
    "id": "m-4444-5555-6666",
    "sequence_number": 1,
    "title": "Borehole Drilling"
  },
  "organization": {
    "id": "o-7777-8888-9999",
    "name": "Water Well Hope NGO",
    "legal_name": "Water Well Hope Foundation Inc"
  },
  "verification_summary": {
    "review_case_id": "rc-0001-aaaa-bbbb",
    "evidence_snapshot_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dual_key_approval": true,
    "dispute_resolved": false
  },
  "disbursement_attestation": null
}
```

---

## 5. Certificate Lifecycle State Machine

```
+-------------------------------------------------------------------------------------------------+
|                                 CERTIFICATE LIFECYCLE TRANSITIONS                               |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   +-----------------------+                                                                     |
|   |  PENDING_ELIGIBILITY  |                                                                     |
|   +-----------------------+                                                                     |
|               |                                                                                 |
|               | Multi-Gate Pass + Ed25519 Signing                                               |
|               v                                                                                 |
|   +-----------------------+           Dispute Investigation                                     |
|   |    ISSUED / ACTIVE    | -----------------------------------------> +---------------------+  |
|   +-----------------------+                                            |      SUSPENDED      |  |
|         |           |       <----------------------------------------- +---------------------+  |
|         |           |                   Investigation Cleared                     |             |
|         |           |                                                             |             |
|         |           | Full Refund / Overturned Review                             |             |
|         |           +---------------------------------------------+               |             |
|         |                                                         |               |             |
|         | Revision Resubmitted                                    v               v             |
|         |                                              +------------------------------------+   |
|         v                                              |              REVOKED               |   |
|   +-----------------------+                            | (Full Refund, Fraud, Overturned)   |   |
|   |      SUPERSEDED       |                            +------------------------------------+   |
|   | (Replaced by New Rev) |                                                                     |
|   +-----------------------+                                                                     |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

### Audit Event Logging
Every transition produces an immutable `CertificateStatusEvent` audit row:
- `certificate_id`: Target certificate.
- `previous_status`: Prior lifecycle state.
- `new_status`: Transitioned lifecycle state.
- `reason_code`: Standard `CertificateRevocationReason` code.
- `reason_description`: Explanation of governance action.
- `actor_id`: User ID of authorizing administrator.
- `created_at`: UTC timestamp.

---

## 6. PDF Rendering & QR Code Specifications

### PDF Document Layout (ReportLab)
- **Dimensions**: Standard A4 Landscape (`297mm x 210mm` / `842pt x 595pt`).
- **Security Border**: Double-line platform frame (`#1E293B` slate blue outer border, `#0D9488` teal inner hairline).
- **Header Elements**: Formal CharityChain crest, certificate title, and unique alphanumeric certificate number.
- **Attestation Statement**: Formal legal certification text detailing recipient name, contribution amount, project, milestone, and NGO.
- **Cryptographic Footprint**:
  - Verification ID (`verification_id`)
  - Key Identifier (`signing_key_id`)
  - Signature Algorithm (`Ed25519`)
  - SHA-256 Payload Hash (`payload_hash`)
  - Truncated signature preview string
- **Interactive QR Code**: 120x120pt high-contrast vector-rendered QR code encoding the public verification URL.
- **SHA-256 Digest**: The rendered PDF binary is hashed with SHA-256 (`pdf_hash`) and stored securely.

