# CharityChain — NGO Verification & Anti-Fraud Architecture

## 1. Overview
CharityChain adheres to the foundational principle:
> **Give with Evidence. Release with Confidence.**

Before an NGO can solicit funds, activate milestones, or appear in public search results, it must undergo formal identity and legal verification by authorized, independent reviewers. This document details the document handling standards, verification queue architecture, anti-conflict-of-interest safeguards, and decision workflows.

---

## 2. Verification Document Specifications

Organizations must upload genuine, verifiable legal documentation before submitting for verification:

| Document Type | Code | Typical Requirement |
|---|---|---|
| Certificate of Incorporation / Registration | `REGISTRATION_CERTIFICATE` | Required. Certificate issued by governing national or state registry. |
| Tax Exemption / Status Determination | `TAX_EXEMPTION_CERTIFICATE` | Required for tax-deductible or exempt institutions (e.g. IRS 501(c)(3)). |
| Non-Profit Status Documentation | `NONPROFIT_STATUS_DOCUMENTATION` | Official charter, public benefit certification. |
| Governing Documents | `GOVERNING_DOCUMENTS` | Bylaws, constitution, board resolution. |
| Physical Address Verification | `ADDRESS_VERIFICATION` | Utility bill, government tax statement, lease. |
| Authorized Representative ID | `AUTHORIZED_REPRESENTATIVE_DOCUMENTATION` | Government photo ID of designated signatory. |
| Other Supporting Documents | `OTHER_SUPPORTING_DOCUMENTS` | Annual audits, impact reports. |

### Document Security & Ingestion Pipeline
Document uploads are processed via `LocalStorageProvider` (`backend/app/integrations/storage/local.py`) with strict defense-in-depth controls:
1. **Magic Bytes Validation**: File contents are inspected for true binary magic signatures:
   - PDF: `%PDF-` (`0x25 0x50 0x44 0x46`)
   - JPEG: `0xFF 0xD8 0xFF`
   - PNG: `\x89PNG\r\n\x1a\n` (`0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A`)
   - Executable, script, and polyglot files (e.g. `MZ`, `ELF`, `<script`) are rejected with `ValidationException`.
2. **File Size Enforcement**: Strict ceiling of 10 MB (`settings.MAX_DOCUMENT_SIZE_MB`).
3. **Cryptographic Content Hashing**: SHA-256 hash calculated over binary content and stored in `OrganizationDocument.file_hash_sha256` for tamper-evidence and deduplication.
4. **Path Traversal Defense**: Filenames are sanitized via regex (`[^a-zA-Z0-9_.-]`), stored with content-addressed prefixes, and normalized using `Path.resolve()` to ensure files remain strictly within the isolated data directory.

---

## 3. Verification Queue Architecture

Reviewers with `PlatformRole.REVIEWER` or `PlatformRole.PLATFORM_ADMIN` access the centralized review queue via `GET /admin/organizations/review-queue`:
- **Query Filter**: Orders pending applications by `submitted_at ASC` (First-In, First-Out priority).
- **Isolation of Reviewers**: Reviewers can inspect full organization legal profiles, download documents, and review historical submission audits.
- **Role Enforcement**: Attempting to query the review queue with a standard `DONOR` or unprivileged role raises `403 Forbidden`.

---

## 4. Conflict of Interest (COI) Protection

A critical governance requirement of CharityChain is preventing self-verification and affiliated reviews:
> **A reviewer must never verify or decide upon an organization to which they belong.**

### Implementation Mechanism:
In `NGOOnboardingService.review_organization` and `ProjectService.review_project`:
```python
reviewer_membership = db.execute(
    select(OrganizationMembership).where(
        and_(
            OrganizationMembership.organization_id == org_id,
            OrganizationMembership.user_id == reviewer_id,
            OrganizationMembership.is_active == True
        )
    )
).scalar_one_or_none()

if reviewer_membership:
    raise ForbiddenException("Conflict of Interest: You cannot review or verify an organization you belong to.")
```
If a user holds `PlatformRole.REVIEWER` but is also an `OWNER`, `ADMIN`, or `MEMBER` of the target organization, the system halts execution and raises `ForbiddenException`.

---

## 5. Reviewer Decisions & Revision Workflows

Reviewers execute decisions via `POST /admin/organizations/{id}/decision`:

### Decision Outcomes:
1. **`APPROVE`**:
   - Organization status transitions to `VERIFIED`.
   - `verified_at` set to current UTC timestamp.
   - `reviewer_id` set to reviewer's UUID.
   - Organization is unlocked for project submission and activation.
2. **`REQUEST_REVISION`**:
   - Organization status transitions to `REVISION_REQUIRED`.
   - Reviewer notes stored in `Organization.revision_notes`.
   - Organization owners receive notification with guidance on missing or deficient documentation.
   - Organization owners upload revised documents and re-submit (transitions back to `PENDING`).
3. **`REJECT`**:
   - Organization status transitions to `REJECTED`.
   - Rejection rationale recorded in `Organization.rejection_reason`.
   - Organization cannot create projects or solicit funds.

---

## 6. Suspension and Reactivation

Platform Administrators maintain governance controls to protect donors against bad actors:
- **Suspension (`POST /admin/organizations/{id}/suspend`)**: Immediately sets status to `SUSPENDED` with mandatory justification. All linked projects are hidden from public discovery.
- **Reactivation (`POST /admin/organizations/{id}/reactivate`)**: Reinstates a resolved organization back to `VERIFIED` status upon corrective actions.
