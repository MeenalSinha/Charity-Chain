# CharityChain — Authorization Matrix & RBAC Specification
**Phase 2 Deliverable — Platform Roles, Organization Scopes & Permission Policies**

---

## 1. Role Architecture & Separation Principle

CharityChain enforces a strict architectural boundary between **Platform-Wide Roles** and **Organization-Scoped Roles**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           PLATFORM LEVEL ROLES                            │
│                                                                           │
│   DONOR                     REVIEWER                 PLATFORM_ADMIN       │
│   (Public / Default)        (Appointed Verifier)     (Platform Superuser) │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      ORGANIZATION LEVEL ROLES (NGO Scoped)                │
│                                                                           │
│   OWNER                     ADMIN                    MEMBER               │
│   (Org Founder / Leader)    (Project Manager)        (Evidence Submitter) │
└───────────────────────────────────────────────────────────────────────────┘
```

### Key Principles
1. **No Implicit Elevation**: An `OWNER` or `ADMIN` of an NGO possesses zero administrative permissions over other organizations and has no platform-level `REVIEWER` or `PLATFORM_ADMIN` privileges.
2. **Multi-Tenancy**: A single user account can hold different roles in multiple organizations (e.g. `OWNER` of Organization A, `MEMBER` of Organization B, and regular `DONOR` platform-wide).
3. **Backend Authorization Verification**: Permissions are never resolved from client requests. Identity is resolved via verified JWT claims; organization authorization is verified against persisted `organization_memberships` records in PostgreSQL.

---

## 2. Comprehensive Permission Matrix

| Resource / Action | Unauthenticated (Guest) | DONOR (Platform) | REVIEWER (Platform) | PLATFORM_ADMIN (Platform) | NGO MEMBER (Org Scoped) | NGO ADMIN (Org Scoped) | NGO OWNER (Org Scoped) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Register & Login** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Manage Own Profile & Sessions** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **View Active Public Projects** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Initiate Donation via PayPal** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **View Own Impact Certificates** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Create NGO Project / Milestone** | ❌ | ❌ | ❌ | ✅ (Override) | ❌ | ✅ (Own Org) | ✅ (Own Org) |
| **Upload Milestone Evidence** | ❌ | ❌ | ❌ | ✅ (Override) | ✅ (Own Org) | ✅ (Own Org) | ✅ (Own Org) |
| **Inspect AI Forensic Indicators** | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ (Own Org) | ✅ (Own Org) |
| **Review & Approve/Reject Milestone**| ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Authorize Milestone Payout** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Invite Members to Organization** | ❌ | ❌ | ❌ | ✅ (Override) | ❌ | ✅ (Admin/Member) | ✅ (Any Role) |
| **Remove Organization Member** | ❌ | ❌ | ❌ | ✅ (Override) | ❌ | ✅ (Member only) | ✅ (Except last Owner) |
| **Change Member Organization Role** | ❌ | ❌ | ❌ | ✅ (Override) | ❌ | ❌ | ✅ (Except last Owner) |
| **List Platform Users & Details** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Suspend / Reactivate Accounts** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Assign Platform Role (Reviewer)** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Inspect Security Audit Events** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |

---

## 3. Invariant & Governance Rules

### 3.1 Organization Last-Owner Protection
- An organization must retain at least one active user with the `OWNER` role at all times.
- Attempts to remove or demote the sole remaining `OWNER` of an organization are rejected with `HTTP 422: Cannot remove or demote the last Owner of an organization. Transfer ownership first.`

### 3.2 Anti-Privilege Escalation Rules
- **No Self-Elevation**: Users cannot modify their own platform role or organization role.
- **Invitation Boundary**: Organization invitations can only grant organization-scoped roles (`OWNER`, `ADMIN`, `MEMBER`). Platform roles cannot be assigned via invitation.
- **Email Recipient Validation**: A user cannot accept an invitation that was addressed to a different email address.
- **IDOR Defense**: All routes modifying or accessing private resources enforce `verify_resource_ownership(resource_owner_id, current_user)`.

