# Phase 8 Database Schema Specification & Migration Documentation

## 1. Migration Overview

- **Migration ID**: `008_phase8_disbursement_and_financial_ledger`
- **Revision File**: `backend/alembic/versions/008_phase8_disbursement_and_financial_ledger.py`
- **Parent Revision**: `007_phase7_human_review_and_decision_workflows`
- **Verification Status**: Tested bidirectionally: `upgrade head` -> `downgrade -1` -> `upgrade head`.

This migration introduces 10 new financial and ledger tables and enhances the core `disbursements` table to support enterprise double-entry accounting, segregated milestone fund releases, secure encrypted payout destinations, multi-item payout batching, and automated reconciliation.

---

## 2. New Database Tables

### 2.1 Double-Entry General Ledger

#### `ledger_accounts`
Stores the Chart of Accounts supporting assets, liabilities, equities, revenues, and expenses.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique account ID |
| `account_number` | `String(32)` | Unique, Not Null, Index | e.g. `10100`, `20100`, `50100` |
| `account_name` | `String(128)` | Not Null | Account description |
| `account_type` | `String(32)` | Not Null | `ASSET`, `LIABILITY`, `EQUITY`, `REVENUE`, `EXPENSE` |
| `currency` | `String(3)` | Not Null, Default `USD` | ISO 4217 code |
| `balance_cents` | `BigInteger` | Not Null, Default `0` | Running balance in minor units |
| `is_active` | `Boolean` | Not Null, Default `True` | Account activity flag |
| `description` | `Text` | Nullable | Optional explanatory notes |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |
| `updated_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `journal_entries`
Records immutable financial transactions balancing debit and credit journal lines.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique entry ID |
| `entry_number` | `String(64)` | Unique, Not Null, Index | e.g. `JE-20261112-0001` |
| `reference_type` | `String(64)` | Nullable | e.g. `DONATION`, `DISBURSEMENT`, `FEE`, `REVERSAL` |
| `reference_id` | `UUID` | Nullable, Index | Entity UUID (donation ID, disbursement ID) |
| `description` | `String(255)` | Not Null | Transaction summary |
| `posted_at` | `DateTime` | Not Null, Default UTC now | Time entry was posted |
| `created_by_id` | `UUID` | Nullable, FK `users.id` | User or system actor |
| `is_reversed` | `Boolean` | Not Null, Default `False` | True if entry has been reversed |
| `reversal_entry_id` | `UUID` | Nullable, FK `journal_entries.id` | Pointer to reversing journal entry |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `journal_lines`
Individual debit and credit postings. Enforces $\sum D == \sum C$.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique line ID |
| `journal_entry_id` | `UUID` | Not Null, FK `journal_entries.id` | Parent journal entry |
| `account_id` | `UUID` | Not Null, FK `ledger_accounts.id` | Target ledger account |
| `entry_type` | `String(16)` | Not Null | `DEBIT` or `CREDIT` |
| `amount_cents` | `BigInteger` | Not Null | Integer minor units (must be $> 0$) |
| `description` | `String(255)` | Nullable | Line description |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

---

### 2.2 Payout Recipient & Batch Infrastructure

#### `payout_recipients`
Maintains verified, encrypted payout credentials for recipient NGOs.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique recipient credential ID |
| `organization_id` | `UUID` | Not Null, FK `organizations.id` | Owning NGO |
| `payout_method` | `String(32)` | Not Null | `PAYPAL`, `BANK_TRANSFER`, `CRYPTO` |
| `encrypted_destination` | `Text` | Not Null | Fernet AES-256 encrypted details |
| `destination_hash` | `String(64)` | Not Null, Index | SHA-256 blind index for lookups |
| `masked_destination` | `String(64)` | Not Null | Sanitized display string (e.g. `d***s@org.com`) |
| `currency` | `String(3)` | Not Null, Default `USD` | ISO 4217 currency |
| `status` | `String(32)` | Not Null, Default `PENDING` | `PENDING`, `VERIFIED`, `REJECTED`, `SUSPENDED` |
| `verification_notes` | `Text` | Nullable | Compliance verification notes |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |
| `updated_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `payout_batches`
Groups approved disbursements for bulk settlement through external gateways.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique batch ID |
| `batch_reference` | `String(64)` | Unique, Not Null, Index | Correlation ID (e.g. `PB-20261112-XYZ`) |
| `provider` | `String(32)` | Not Null | `PAYPAL`, `STRIPE_CONNECT`, `MOCK` |
| `status` | `String(32)` | Not Null, Default `PENDING` | `PENDING`, `PROCESSING`, `COMPLETED`, `PARTIALLY_COMPLETED`, `FAILED` |
| `total_amount_cents` | `BigInteger` | Not Null, Default `0` | Sum of batch disbursement amounts |
| `fee_amount_cents` | `BigInteger` | Not Null, Default `0` | Incurred gateway fees |
| `currency` | `String(3)` | Not Null, Default `USD` | ISO 4217 currency |
| `disbursement_count` | `Integer` | Not Null, Default `0` | Number of items in batch |
| `submitted_at` | `DateTime` | Nullable | Gateway transmission timestamp |
| `completed_at` | `DateTime` | Nullable | Final settlement timestamp |
| `error_message` | `Text` | Nullable | Provider error message if failed |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |
| `updated_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `payout_approvals`
Maker-Checker audit log tracking approvals and rejections of disbursements.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique approval ID |
| `disbursement_id` | `UUID` | Not Null, FK `disbursements.id` | Target disbursement |
| `approver_id` | `UUID` | Not Null, FK `users.id` | Approving financial officer |
| `action` | `String(32)` | Not Null | `APPROVE` or `REJECT` |
| `notes` | `Text` | Nullable | Reason / review notes |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

---

### 2.3 Automated Reconciliation Infrastructure

#### `reconciliation_runs`
Tracks scheduled and manual reconciliation batch executions.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique run ID |
| `run_reference` | `String(64)` | Unique, Not Null, Index | e.g. `REC-20261112-001` |
| `provider` | `String(32)` | Not Null | `PAYPAL`, `BANK` |
| `window_start` | `DateTime` | Not Null | Start of reconciliation period |
| `window_end` | `DateTime` | Not Null | End of reconciliation period |
| `status` | `String(32)` | Not Null | `PENDING`, `IN_PROGRESS`, `COMPLETED`, `DISCREPANCY_FLAGGED` |
| `total_records_analyzed`| `Integer` | Not Null, Default `0` | Records processed |
| `matched_count` | `Integer` | Not Null, Default `0` | Clean matches |
| `discrepancy_count` | `Integer` | Not Null, Default `0` | Variance count |
| `total_unmatched_amount_cents` | `BigInteger` | Not Null, Default `0` | Discrepancy total |
| `currency` | `String(3)` | Not Null, Default `USD` | ISO 4217 currency |
| `notes` | `Text` | Nullable | Run observations |
| `executed_by_id` | `UUID` | Nullable, FK `users.id` | Actor who triggered run |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |
| `updated_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `reconciliation_items`
Detailed line-item matching results and resolution audit trail.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | Primary Key, default UUIDv4 | Unique item ID |
| `run_id` | `UUID` | Not Null, FK `reconciliation_runs.id` | Parent run |
| `item_type` | `String(32)` | Not Null | `DONATION`, `DISBURSEMENT`, `FEE` |
| `internal_reference_id`| `UUID` | Nullable, Index | Internal entity ID |
| `external_transaction_id`| `String(128)` | Nullable, Index | Gateway transaction ID |
| `expected_amount_cents`| `BigInteger` | Not Null | Internal record amount |
| `actual_amount_cents` | `BigInteger` | Not Null | Gateway settled amount |
| `fee_variance_cents` | `BigInteger` | Not Null, Default `0` | Fee discrepancy |
| `status` | `String(32)` | Not Null | `MATCHED`, `DISCREPANCY`, `RESOLVED` |
| `discrepancy_type` | `String(32)` | Nullable | e.g. `AMOUNT_MISMATCH`, `FEE_MISMATCH` |
| `resolution_notes` | `Text` | Nullable | Auditor explanation |
| `resolved_by_id` | `UUID` | Nullable, FK `users.id` | Resolving officer |
| `resolved_at` | `DateTime` | Nullable | Timestamp of resolution |
| `created_at` | `DateTime` | Not Null, Default UTC now | Timestamp |
| `updated_at` | `DateTime` | Not Null, Default UTC now | Timestamp |

#### `settlement_records` & `processor_fee_records`
Ingested raw settlement entries and itemized processing fees from external gateways.

---

## 3. Alterations to Existing Tables

### `disbursements`
Enhanced to support maker-checker approvals, recipient linking, batch scheduling, and retry tracking:

- Added `payout_batch_id`: Foreign key to `payout_batches.id` (nullable).
- Added `payout_recipient_id`: Foreign key to `payout_recipients.id` (nullable).
- Added `requested_by_id`: Foreign key to `users.id` (nullable).
- Added `approved_by_id`: Foreign key to `users.id` (nullable).
- Added `approval_date`: Timestamp of checker signoff (nullable).
- Added `batch_item_id`: Gateway batch item reference (nullable).
- Added `failure_reason`: Error description upon failure (nullable).
- Added `retry_count`: Integer retry attempts counter (default `0`).
- Added `last_retry_at`: Timestamp of latest retry attempt (nullable).
