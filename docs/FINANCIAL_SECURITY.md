# Financial Security, Controls & Threat Mitigation Architecture

## 1. Executive Summary

Financial operations in CharityChain handle donor contributions, project escrow holdings, and outbound vendor/NGO payouts. The financial architecture implements defense-in-depth security principles designed to prevent unauthorized fund diversion, insider collusion, data tampering, replay attacks, and sensitive data leakage.

Every financial operation is governed by strict mathematical invariants, role segregation, cryptographic controls at rest, and auditable state transitions.

---

## 2. Threat Model & Mitigation Matrix

| Threat Vector | Attack Scenario | Architectural Mitigation | Enforcement Layer |
| :--- | :--- | :--- | :--- |
| **Insider Embezzlement** | A platform operator or NGO staff member creates and approves a disbursement to an unauthorized account. | **Maker-Checker Segregation**: Requester cannot approve their own disbursement. Two independent authenticated actors required. | `DisbursementService` & `DisbursementEligibilityService` |
| **Insecure Direct Object Reference (IDOR)** | An attacker modifies payout recipient details of another organization or inspects other donors' payments. | **Tenant & Organization Boundary Checks**: Recipient, payment, and disbursement endpoints enforce organization/user ownership. | API Routers (`/payout-recipients`, `/disbursements`, `/payments`) |
| **Replay / Duplicate Payouts** | Network retries or malicious repeated requests trigger duplicate disbursements for a single milestone. | **Idempotency Keys & State Machine Guards**: Batches and disbursements use UUID correlation tokens and strict terminal state validation. | `PayoutBatch`, `Disbursement` state machine |
| **Ghost Settlement / Phantom Funds** | Fallback mock or synthetic capture IDs bypass actual payment gateway settlement verification. | **Elimination of Synthetic Fallbacks**: Only genuine `COMPLETED` gateway captures adjust escrow; failures raise hard exceptions. | `DonationService` & `PayPalPayoutProvider` |
| **Data Breach / Credential Exposure** | Attacker accesses SQL database directly and extracts recipient bank account numbers or PayPal emails. | **AES/Fernet Encryption at Rest & Blind Indexing**: Sensitive destinations are encrypted; lookups use SHA-256 blind hashes. | `PayoutRecipient` model & `PayoutRecipientService` |
| **Ledger Tampering / Accounting Fraud** | An operator attempts to alter past balances or delete transaction records to conceal missing funds. | **Ledger Immutability & Double-Entry Invariant**: Updates and deletions prohibited; entries must balance ($\sum D = \sum C$); corrections require offsetting reversals. | `LedgerService` & Database constraints |
| **Over-Disbursement** | A disbursement is requested that exceeds the net funds contributed or milestone budget ceiling. | **Balance Ceilings & Pre-Flight Gate Audits**: Available balance dynamically calculated as $C_i - (D_i + P_i)$; checked before approval. | `FundAllocationService` & Gate 5 check |

---

## 3. Maker-Checker Segregation (Four-Eyes Principle)

To prevent unilateral disbursement of funds, CharityChain enforces a strict Maker-Checker model:

```
[ Maker: NGO Representative / Admin ]
                 |
                 v
   Create Disbursement Request
   (Status: REQUESTED)
                 |
                 |---------> Cannot Self-Approve!
                 |           (If requester_id == approver_id -> 403 Forbidden)
                 v
[ Checker: Financial Officer / Independent Admin ]
                 |
                 v
   Review Eligibility Gate Checks & Balances
                 |
                 v
   Approve Disbursement Request
   (Status: APPROVED)
                 |
                 v
[ Batch Execution Engine / Automated Worker ]
                 |
                 v
   Submit to Payout Gateway (PayPal / Bank)
   (Status: PROCESSING -> COMPLETED)
```

### Invariant Rules:
1. `disbursement.requested_by_id != current_user.id`: An operator cannot approve any disbursement they created.
2. Only users with `ADMIN` or `FINANCIAL_OFFICER` roles can perform checker approvals.
3. Once approved, the record links the `approved_by_id`, `approval_timestamp`, and an immutable cryptographic signature/hash of the approval payload.

---

## 4. Cryptographic Data Protection at Rest

### 4.1 AES/Fernet Encryption
All sensitive recipient destinations (bank account numbers, routing numbers, PayPal account email addresses) are encrypted prior to database persistence using AES in CBC mode with HMAC-SHA256 authenticated encryption (`cryptography.fernet.Fernet`):

```python
# Encryption workflow in PayoutRecipientService
cipher = Fernet(settings.ENCRYPTION_KEY)
encrypted_destination = cipher.encrypt(raw_destination.encode("utf-8")).decode("utf-8")
```

### 4.2 Blind Indexing (Deterministic Search Without Decryption)
To query whether a payout destination is already registered to an organization (preventing duplicates) without decrypting every row in the table, CharityChain maintains a deterministic SHA-256 blind index:

$$\text{destination\_hash} = \text{HMAC-SHA256}(\text{blind\_salt}, \text{normalized\_destination})$$

The database indexes `destination_hash`, enabling $O(1)$ lookups while ensuring the underlying destination remains encrypted at rest.

### 4.3 Response Masking
Sensitive destinations are masked before being returned in API responses:
- Email: `d***s@cleanwaters.org`
- Bank Account: `************6789`
- Routing Number: `*****123`

---

## 5. Idempotency & Duplicate Payout Prevention

Outbound financial transfers must be idempotent across network drops, timeouts, and concurrent requests:

1. **Batch Reference Identifiers**: Each payout batch generates a cryptographically random, deterministic reference ID (`PB-{timestamp}-{uuid}`).
2. **Item Correlation IDs**: Each disbursement line in a batch carries an immutable correlation ID (`DISB-{disbursement_id}`).
3. **Database Concurrency Protection**:
   - `SELECT FOR UPDATE` row-level locks on milestone and disbursement records prevent concurrent double-spend.
   - Transitioning a disbursement to `BATCHED` or `PROCESSING` is atomic with batch creation.
4. **Gateway Idempotency**:
   - Payout requests to payment providers include provider-native idempotency keys (e.g., PayPal's `sender_batch_header.sender_batch_id`). If the provider receives a duplicate call with the same sender batch ID, it returns the existing batch status rather than re-executing the payout.

---

## 6. Double-Entry Ledger Mathematical Verification

The double-entry ledger enforces mathematical guarantees across all transactions:

1. **Balance Invariant**:
   Every journal entry must contain at least two lines, and the sum of all debits must exactly equal the sum of all credits:
   $$\sum \text{Debit Amounts} == \sum \text{Credit Amounts}$$
2. **Zero Modification Policy**:
   - `UPDATE` and `DELETE` operations on `journal_entries` and `journal_lines` are rejected by application logic.
   - When a correction or refund occurs, an **offsetting reversal entry** is posted, referencing the original entry ID and documenting the reason.
3. **Continuous Trial Balance**:
   The ledger supports real-time trial balance queries across asset, liability, equity, revenue, and expense accounts:
   $$\sum \text{All Account Debits} - \sum \text{All Account Credits} = 0$$

---

## 7. Security Verification Test Suites

CharityChain's security posture is continually verified through dedicated test suites:
- `tests/security/test_financial_maker_checker.py`: Verifies self-approval rejection and role segregation.
- `tests/security/test_duplicate_payout_prevention.py`: Verifies idempotency, concurrency locks, and replay prevention.
- `tests/security/test_ledger_immutability.py`: Verifies mathematical balance invariants and anti-tampering rules.
- `tests/security/test_payment_security.py`: Verifies IDOR protection, authentication requirements, and encryption.
