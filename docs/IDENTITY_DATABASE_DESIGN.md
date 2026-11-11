# CharityChain — Identity Database Design Specification
**Phase 2 Deliverable — Schema Extensions, Token Models, Constraints & Alembic Migration**

---

## 1. Identity & Session ER Diagram

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : maintains
    USERS ||--o{ EMAIL_VERIFICATION_TOKENS : receives
    USERS ||--o{ PASSWORD_RESET_TOKENS : requests
    USERS ||--o{ ORGANIZATION_MEMBERSHIPS : holds
    USERS ||--o{ ORGANIZATION_INVITATIONS : sent_by

    ORGANIZATIONS ||--o{ ORGANIZATION_MEMBERSHIPS : includes
    ORGANIZATIONS ||--o{ ORGANIZATION_INVITATIONS : targeted_for

    USERS {
        string id PK
        string email UK "Normalized lowercase"
        string hashed_password "Argon2id digest"
        string full_name
        string role "PlatformRole (DONOR, REVIEWER, PLATFORM_ADMIN)"
        string account_status "AccountStatus enum"
        boolean is_verified
        datetime email_verified_at
        datetime password_changed_at
        datetime last_login_at
        int failed_login_attempts
        datetime locked_until
    }

    USER_SESSIONS {
        string id PK
        string user_id FK
        string refresh_token_hash UK "SHA-256 digest"
        string token_family "UUID grouping rotated tokens"
        string ip_address
        string user_agent
        datetime expires_at
        datetime last_used_at
        boolean is_revoked
        datetime revoked_at
        string revocation_reason
    }

    EMAIL_VERIFICATION_TOKENS {
        string id PK
        string user_id FK
        string token_hash UK "SHA-256 digest"
        datetime expires_at
        datetime used_at
        boolean is_used
    }

    PASSWORD_RESET_TOKENS {
        string id PK
        string user_id FK
        string token_hash UK "SHA-256 digest"
        datetime expires_at
        datetime used_at
        boolean is_used
    }

    ORGANIZATION_MEMBERSHIPS {
        string id PK
        string user_id FK
        string organization_id FK
        string role "OrganizationRole (OWNER, ADMIN, MEMBER)"
        boolean is_active
        datetime created_at
    }

    ORGANIZATION_INVITATIONS {
        string id PK
        string organization_id FK
        string inviter_id FK
        string invitee_email
        string role "OrganizationRole"
        string token_hash UK "SHA-256 digest"
        datetime expires_at
        datetime accepted_at
        boolean is_revoked
        string status "InvitationStatus enum"
    }
```

---

## 2. Table Definitions & Constraint Rules

### 2.1 Table: `users` (Extended)
- `email`: Normalized lowercase string, indexed with unique B-tree index.
- `role`: PlatformRole enum (`DONOR`, `REVIEWER`, `PLATFORM_ADMIN`).
- `account_status`: AccountStatus enum (`PENDING_VERIFICATION`, `ACTIVE`, `LOCKED`, `SUSPENDED`, `DEACTIVATED`).
- `failed_login_attempts`: Integer counter tracking consecutive bad passwords.
- `locked_until`: Expiration timestamp for brute-force lockouts.

### 2.2 Table: `user_sessions`
- `refresh_token_hash`: Unique SHA-256 hash string (`VARCHAR(64)`). Plaintext refresh tokens are never persisted.
- `token_family`: Grouping identifier enabling complete family revocation upon replay attack detection.
- `is_revoked`: Boolean flag updated on rotation, logout, or security invalidation.

### 2.3 Table: `organization_memberships`
- Composite Unique Constraint: `UniqueConstraint('user_id', 'organization_id', name='uq_org_membership_user_org')` prevents duplicate memberships.
- `role`: OrganizationRole enum (`OWNER`, `ADMIN`, `MEMBER`).

### 2.4 Table: `organization_invitations`
- `token_hash`: Unique SHA-256 digest of URL-safe random string.
- Foreign Keys: Cascades on organization or inviter user deletion.

---

## 3. Alembic Forward Migration

The changes are captured in [`backend/alembic/versions/002_phase2_identity_and_auth.py`](file:///c:/Users/Meenal%20Sinha/Downloads/Charity-Chain/backend/alembic/versions/002_phase2_identity_and_auth.py):
- Adds columns to `users` with non-null server defaults.
- Adds `uq_org_membership_user_org` unique constraint.
- Creates `user_sessions`, `email_verification_tokens`, `password_reset_tokens`, and `organization_invitations` with appropriate foreign keys and performance indexes.
- Implements fully reversible `downgrade()` logic.

