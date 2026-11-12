# CharityChain — Phase 10 Database Changes & Schema Migration Report

## 1. Migration Overview

- **Migration ID**: `010_phase10_notifications_and_webhooks`
- **Revises**: `009_phase9_impact_ledger_and_certificates`
- **Target Subsystem**: Transactional Outbox, In-App Notifications, User Preferences, Outbound Webhooks, and Delivery Tracking.
- **Verification**: Reversible migration tested with `alembic upgrade head` and verified against SQLAlchemy models.

---

## 2. Newly Created Tables

### 1. `notification_outbox`
Transactional outbox ensuring domain state changes and notification events are committed atomically.
- `id` (VARCHAR(36), PK)
- `event_id` (VARCHAR(64), UNIQUE, INDEX)
- `event_type` (VARCHAR(64), INDEX)
- `entity_type` (VARCHAR(64))
- `entity_id` (VARCHAR(64))
- `organization_id` (VARCHAR(36), NULLABLE, INDEX)
- `actor_user_id` (VARCHAR(36), NULLABLE)
- `correlation_id` (VARCHAR(64), NULLABLE)
- `idempotency_key` (VARCHAR(128), UNIQUE, INDEX)
- `payload_json` (TEXT)
- `status` (ENUM('PENDING', 'PROCESSING', 'PROCESSED', 'FAILED', 'DEAD_LETTER'), INDEX)
- `retry_count` (INTEGER, DEFAULT 0)
- `max_retries` (INTEGER, DEFAULT 5)
- `next_retry_at` (DATETIME WITH TIMEZONE, INDEX)
- `error_message` (TEXT, NULLABLE)
- `processed_at` (DATETIME WITH TIMEZONE, NULLABLE)
- `created_at` (DATETIME, NOT NULL)
- `updated_at` (DATETIME, NOT NULL)

### 2. `notifications`
In-app notification center alerts for individual donors, NGO staff, and reviewers.
- `id` (VARCHAR(36), PK)
- `user_id` (VARCHAR(36), FK `users.id`, INDEX)
- `organization_id` (VARCHAR(36), FK `organizations.id`, NULLABLE, INDEX)
- `category` (ENUM('DONATIONS', 'PAYMENTS', 'PROJECTS', 'MILESTONES', 'EVIDENCE', 'VERIFICATION', 'DISBURSEMENTS', 'CERTIFICATES', 'SECURITY', 'SYSTEM', 'ACCOUNT'), INDEX)
- `event_type` (VARCHAR(64), INDEX)
- `title` (VARCHAR(255))
- `message` (TEXT)
- `action_url` (VARCHAR(512), NULLABLE)
- `entity_type` (VARCHAR(64), NULLABLE)
- `entity_id` (VARCHAR(64), NULLABLE)
- `is_read` (BOOLEAN, DEFAULT FALSE, INDEX)
- `read_at` (DATETIME WITH TIMEZONE, NULLABLE)
- `is_archived` (BOOLEAN, DEFAULT FALSE, INDEX)
- `archived_at` (DATETIME WITH TIMEZONE, NULLABLE)

### 3. `notification_preferences`
User communication channel and topic settings.
- `id` (VARCHAR(36), PK)
- `user_id` (VARCHAR(36), FK `users.id`, UNIQUE, INDEX)
- `email_enabled` (BOOLEAN, DEFAULT TRUE)
- `in_app_enabled` (BOOLEAN, DEFAULT TRUE)
- `donation_updates` (BOOLEAN, DEFAULT TRUE)
- `project_updates` (BOOLEAN, DEFAULT TRUE)
- `milestone_updates` (BOOLEAN, DEFAULT TRUE)
- `verification_updates` (BOOLEAN, DEFAULT TRUE)
- `certificate_notifications` (BOOLEAN, DEFAULT TRUE)
- `security_alerts` (BOOLEAN, DEFAULT TRUE, NOT NULL)

### 4. `notification_delivery_attempts`
Telemetry records for outbound email dispatches.
- `id` (VARCHAR(36), PK)
- `notification_id` (VARCHAR(36), FK `notifications.id`, NULLABLE, INDEX)
- `channel` (ENUM('IN_APP', 'EMAIL'))
- `recipient` (VARCHAR(255))
- `status` (VARCHAR(32), INDEX)
- `attempt_number` (INTEGER, DEFAULT 1)
- `response_code` (VARCHAR(32), NULLABLE)
- `error_message` (TEXT, NULLABLE)
- `attempted_at` (DATETIME WITH TIMEZONE, DEFAULT UTC NOW)

### 5. `webhook_subscriptions`
External webhook endpoint subscriptions.
- `id` (VARCHAR(36), PK)
- `organization_id` (VARCHAR(36), FK `organizations.id`, NULLABLE, INDEX)
- `created_by_user_id` (VARCHAR(36), FK `users.id`, NOT NULL, INDEX)
- `endpoint_url` (VARCHAR(512), NOT NULL)
- `description` (VARCHAR(255), NULLABLE)
- `subscribed_events` (JSON, DEFAULT '[]')
- `consecutive_failures` (INTEGER, DEFAULT 0)
- `suspended_at` (DATETIME WITH TIMEZONE, NULLABLE)

### 6. `webhook_endpoint_secrets`
Fernet-encrypted HMAC signing secrets.
- `id` (VARCHAR(36), PK)
- `subscription_id` (VARCHAR(36), FK `webhook_subscriptions.id`, INDEX)
- `secret_encrypted` (TEXT, NOT NULL)
- `secret_prefix` (VARCHAR(16), NOT NULL)
- `rotated_at` (DATETIME WITH TIMEZONE, NULLABLE)

### 7. `webhook_deliveries`
Event dispatch records per webhook subscription.
- `id` (VARCHAR(36), PK)
- `subscription_id` (VARCHAR(36), FK `webhook_subscriptions.id`, INDEX)
- `event_id` (VARCHAR(64), INDEX)
- `event_type` (VARCHAR(64), INDEX)
- `payload_json` (TEXT)
- `status` (ENUM('PENDING', 'DELIVERED', 'FAILED', 'DEAD_LETTER'), INDEX)
- `retry_count` (INTEGER, DEFAULT 0)
- `max_retries` (INTEGER, DEFAULT 5)
- `next_retry_at` (DATETIME WITH TIMEZONE, INDEX)
- `completed_at` (DATETIME WITH TIMEZONE, NULLABLE)

### 8. `webhook_delivery_attempts`
Per-attempt HTTP response telemetry.
- `id` (VARCHAR(36), PK)
- `delivery_id` (VARCHAR(36), FK `webhook_deliveries.id`, INDEX)
- `attempt_number` (INTEGER, NOT NULL)
- `status_code` (INTEGER, NULLABLE)
- `response_headers_json` (TEXT, NULLABLE)
- `response_body` (TEXT, NULLABLE)
- `error_message` (TEXT, NULLABLE)
- `execution_duration_ms` (INTEGER, DEFAULT 0)
- `attempted_at` (DATETIME WITH TIMEZONE, DEFAULT UTC NOW)

### 9. `notification_template_versions`
Versioned email and notification layout content.
- `id` (VARCHAR(36), PK)
- `template_key` (VARCHAR(64), INDEX)
- `version` (VARCHAR(16), INDEX)
- `subject_template` (VARCHAR(255))
- `html_template` (TEXT)
- `text_template` (TEXT)
- `is_active` (BOOLEAN, DEFAULT TRUE)

