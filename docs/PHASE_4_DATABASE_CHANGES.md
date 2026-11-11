# Phase 4 Database Schema Changes & Migration Report

## 1. Alembic Migration Script

- **Migration Version**: `004_phase4_payment_and_donations`
- **File**: `backend/alembic/versions/004_phase4_payment_and_donations.py`
- **Down Revision**: `003_phase3_ngo_and_projects`

---

## 2. Table Schema Modifications

### A. `donations` Table Updates
Added columns to track project relationships, donor identities, and completion timestamps:
- `project_id`: `VARCHAR(36)`, Foreign Key to `projects.id` (`ondelete="CASCADE"`), `nullable=False`, indexed.
- `status`: `VARCHAR(50)`, default `'CREATED'`, `nullable=False`, indexed.
- `donor_name`: `VARCHAR(255)`, nullable.
- `donor_email`: `VARCHAR(255)`, nullable.
- `is_anonymous`: `BOOLEAN`, default `False`, `nullable=False`.
- `completed_at`: `DATETIME`, nullable.
- `cancelled_at`: `DATETIME`, nullable.

### B. `payment_orders` Table Updates
Enhanced with idempotency keys, approval redirection URLs, and error diagnostics:
- `idempotency_key`: `VARCHAR(100)`, nullable, indexed.
- `approval_url`: `VARCHAR(512)`, nullable.
- `failure_code`: `VARCHAR(100)`, nullable.
- `failure_reason`: `TEXT`, nullable.

### C. `payment_captures` Table Updates
Enhanced with order relationship linking and settlement timestamps:
- `order_id`: `VARCHAR(36)`, Foreign Key to `payment_orders.id` (`ondelete="CASCADE"`), `nullable=False`, indexed.
- `idempotency_key`: `VARCHAR(100)`, nullable, indexed.
- `captured_at`: `DATETIME`, nullable.

### D. `payment_refunds` Table Updates
Enhanced with idempotency tracking and refund execution timestamps:
- `idempotency_key`: `VARCHAR(100)`, nullable, indexed.
- `refunded_at`: `DATETIME`, nullable.

### E. New `webhook_events` Table
Created to provide event deduplication and cryptographic audit trails for incoming provider notifications:
```sql
CREATE TABLE webhook_events (
    id VARCHAR(36) PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    event_id VARCHAR(100) NOT NULL UNIQUE,
    event_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(100),
    resource_type VARCHAR(100),
    signature_verification_status VARCHAR(50) NOT NULL,
    processing_status VARCHAR(50) NOT NULL,
    payload TEXT,
    failure_reason TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1
);
CREATE INDEX ix_webhook_events_event_id ON webhook_events (event_id);
CREATE INDEX ix_webhook_events_event_type ON webhook_events (event_type);
```

---

## 3. Rollback & Reversibility Verification

The migration includes a complete `downgrade()` implementation that:
1. Drops the `webhook_events` table and its indexes.
2. Removes the added columns from `payment_refunds`, `payment_captures`, `payment_orders`, and `donations`.
Both forward migration and reverse downgrade are validated.
