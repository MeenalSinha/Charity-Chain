# CharityChain — Review Audit Trail & Cryptographic Verification

## 1. Overview & Tamper-Evident Architecture

CharityChain's human review decisions govern public trust and future financial releases. To provide mathematical proof of integrity, all actions taken within Phase 7 are recorded in an append-only, tamper-evident audit ledger (`review_audit_events`).

Every event is cryptographically chained to its predecessor using SHA-256 hashes, ensuring that retroactive deletion, modification, or reordering of audit events is computationally detectable.

---

## 2. Hash Chaining Formula

For each new audit event $i$, its hash is computed deterministically:

$$\text{CanonicalDetails}_i = \text{JSON\_DUMP}(\text{details}_i, \text{sort\_keys}=\text{True}, \text{separators}=(\text{","}, \text{":"}))$$

$$\text{Payload}_i = \text{PrevHash}_{i-1} \parallel \text{MilestoneID}_i \parallel \text{ActorID}_i \parallel \text{EventType}_i \parallel \text{CanonicalDetails}_i$$

$$\text{CurrHash}_i = \text{SHA-256}(\text{Payload}_i)$$

For the initial event in a milestone's lifecycle ($i=0$), $\text{PrevHash}_0$ is defined as 64 zeros:
$$\text{PrevHash}_0 = \underbrace{\text{"00000000...0000"}}_{\text{64 zeros}}$$

---

## 3. Audited Event Types

The audit engine tracks eleven discrete lifecycle event types:

| Event Type | Trigger | Key Logged Attributes |
|---|---|---|
| `CASE_CREATED` | NGO submits milestone for review. | `review_case_id`, `submission_round`, `snapshot_digest`, `evidence_count` |
| `REVIEWERS_ASSIGNED` | Administrator assigns dual reviewers. | `reviewer_1_id`, `reviewer_2_id`, `task_ids` |
| `ASSIGNMENT_ACCEPTED` | Reviewer accepts assigned task. | `task_id`, `reviewer_slot` |
| `ASSIGNMENT_DECLINED` | Reviewer declines task with reason. | `task_id`, `reviewer_slot`, `reason` |
| `CONFLICT_DECLARED` | Reviewer declares conflict of interest. | `task_id`, `reviewer_slot`, `reason` |
| `DECISION_SUBMITTED` | Reviewer records independent decision. | `task_id`, `slot`, `decision`, truncated `comments` |
| `DECISION_RECONCILED` | Engine reconciles dual decisions. | `result`, `decisions_count`, `milestone_status` |
| `ESCALATED` | Split decision triggers escalation. | `result`, `decisions`, escalation message |
| `ADJUDICATED` | Admin resolves split decision. | `adjudicator_id`, `final_decision`, `comments` |
| `REVISION_REQUESTED` | Concurring revision requirements issued. | `result`, `deadline`, structured requirements |
| `NGO_RESUBMITTED` | NGO submits round N+1 evidence. | `prior_case_id`, `new_case_id`, `round`, `evidence_count` |
| `DISPUTE_RAISED` | NGO files formal appeal. | `dispute_id`, `category`, `reason` |
| `DISPUTE_RESOLVED` | Adjudicator resolves appeal. | `dispute_id`, `resolution`, `explanation` |

---

## 4. Chain Verification & Tamper Detection

The system exposes `/api/v1/reviews/audit/milestone/{milestone_id}/verify` to verify the audit chain.

The algorithm:
1. Queries all `ReviewAuditEvent` records for the milestone sorted chronologically by `created_at`.
2. Iterates sequentially from index 0:
   - Validates that `ev.previous_event_hash == current_pointer`.
   - Recomputes the expected hash using `compute_review_audit_hash`.
   - Validates that `ev.current_event_hash == expected_hash`.
   - Advances `current_pointer = ev.current_event_hash`.
3. If any record fails matching or order, `is_chain_intact` returns `False` immediately, pinpointing the corrupted record.
