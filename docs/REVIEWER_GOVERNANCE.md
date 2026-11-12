# CharityChain — Reviewer Governance & Conflict-of-Interest Policy

## 1. Overview & Objective

Reviewer integrity is the cornerstone of CharityChain's verification system. Without strict independence, verification degrades into rubber-stamping or biased adjudication. This document codifies reviewer eligibility criteria, mandatory conflict-of-interest (COI) prohibitions, workload management, and operational conduct.

---

## 2. Reviewer Eligibility & Account Requirements

A user may only be assigned review tasks or act as an independent reviewer if they fulfill all the following requirements:
1. **Account Verification**: Account must have a verified email address (`is_verified = True`).
2. **Active Account Status**: User status must be `AccountStatus.ACTIVE`. Suspended, flagged, or deactivated accounts cannot review cases.
3. **Authorized Role**: User must possess either `PlatformRole.REVIEWER` or `PlatformRole.PLATFORM_ADMIN`. Regular donors (`PlatformRole.DONOR`) cannot review cases.
4. **Independent Identity**: Reviewers must maintain verified identity profiles on the platform.

---

## 3. Strict Conflict-of-Interest (COI) Rules

The CharityChain review engine programmatically enforces four zero-tolerance COI checks prior to assignment and re-verifies them at decision submission:

| Rule | Description | Programmatic Enforcement |
|---|---|---|
| **COI Rule 1: Organization Affiliation** | A reviewer cannot be an active owner, administrator, or staff member of the organization submitting the milestone. | `OrganizationMembership.organization_id == case.organization_id AND is_active == True` |
| **COI Rule 2: Project Creation** | A reviewer cannot be the creator or lead contributor of the underlying project. | `Project.creator_id == reviewer.id` |
| **COI Rule 3: Milestone Submitter** | A reviewer cannot be the user who submitted the milestone review case. | `ReviewCase.submitted_by_id == reviewer.id` |
| **COI Rule 4: Evidence Contributor** | A reviewer cannot have uploaded any of the evidence files included in the case package. | `EvidenceFile.id IN (case_evidence_ids) AND uploader_id == reviewer.id` |
| **COI Rule 5: Cross-Slot Independence** | The same reviewer cannot occupy both Slot 1 and Slot 2 in the same review case round. | `reviewer_1_id != reviewer_2_id` |
| **COI Rule 6: Dispute Adjudication** | A user who participated in the original review decision cannot adjudicate a formal appeal on that case. | `adjudicator_id NOT IN (prior_reviewer_ids)` |

If any check fails, assignment is rejected with a documented error code (`CONFLICT_OF_INTEREST_*`).

---

## 4. Workload Management & Quota Limits

To ensure high-quality, unhurried evaluations, the system enforces strict workload ceilings:
- **Maximum Open Tasks**: Configured via `REVIEW_MAX_OPEN_TASKS_PER_REVIEWER` (default: 5). A reviewer with 5 active tasks (`ASSIGNED`, `ACCEPTED`, or `IN_PROGRESS`) cannot receive additional task assignments until existing tasks are decided or declined.
- **Task Due Horizon**: Configured via `REVIEW_TASK_DUE_HOURS` (default: 72 hours). Tasks not accepted or decided within this window are flagged for reassignment.
- **Maximum Reassignments**: Configured via `REVIEW_MAX_REASSIGNMENTS` (default: 3). If a task slot is declined or expired three consecutive times, the case is automatically flagged for platform administrator triage.

---

## 5. Voluntary Recusal & Conflict Declaration

If a reviewer discovers a personal, financial, or organizational relationship with a project that is not automatically captured in the database (e.g., family relationship with an NGO director, former employment), the reviewer is ethically and procedurally mandated to declare a conflict of interest via `/api/v1/reviews/tasks/{task_id}/conflict`.

Declaring a conflict:
1. Immediately unassigns the reviewer from the task slot.
2. Transitions the task state to `DECLINED` with `conflict_declared = True`.
3. Records the detailed reason in the immutable audit trail.
4. Frees the slot for administrative reassignment without penalty to the reviewer's rating.

