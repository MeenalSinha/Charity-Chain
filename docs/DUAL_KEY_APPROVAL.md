# CharityChain — Dual-Key Approval Architecture & Security Model

## 1. Security Rationale

Single-actor approval workflows present an inherent vulnerability in grant-making and charitable disbursement systems:
1. **Collusion Vulnerability**: An external bad actor could compromise or bribe a single reviewer to approve fraudulent evidence.
2. **Cognitive Fatigue & Errors**: A single reviewer may overlook sophisticated image doctoring, misleading camera perspectives, or duplicate receipts.
3. **Internal Bias**: Subjective standards between individual reviewers lead to inconsistent evaluations across projects.

CharityChain's **Dual-Key Approval Model** eliminates single points of failure by mandating that milestone completions require unanimous consent from two independent, verified reviewers operating without cross-influence.

---

## 2. Independent Slots & Blind Review Protocol

Every review case consists of two discrete slots:
- **Slot 1 (Primary)**: Evaluates evidence completeness, physical metrics, and AI verification reports.
- **Slot 2 (Secondary)**: Independently repeats evaluation on the exact frozen evidence snapshot.

### Cognitive Anchoring Elimination (Blind Review)
In traditional collaborative systems, seeing a peer's high rating or positive comments primes subsequent reviewers to agree (the *anchoring heuristic*). 

To enforce strict intellectual independence:
1. When Reviewer 1 submits their decision, the decision record is persisted, but the case remains in `AWAITING_SECOND_REVIEW`.
2. When Reviewer 2 views the case details prior to submitting their own decision, Reviewer 1's decision is masked:
   - `reviewer_id` is replaced with `REDACTED_BLIND_REVIEW`.
   - `decision` is replaced with `SUBMITTED_HIDDEN`.
   - `comments` are replaced with a standardized placeholder.
3. Only after Reviewer 2 submits their independent decision are both records unmasked for mutual visibility and platform audit.

---

## 3. Decision Reconciliation Matrix

When both review slots have been decided, the engine executes deterministic reconciliation:

| Reviewer 1 Decision | Reviewer 2 Decision | Reconciled Result | Case Status | Milestone Status | Action Required |
|---|---|---|---|---|---|
| `APPROVE` | `APPROVE` | `CONCURRING_APPROVE` | `APPROVED` | `COMPLETED` | Milestone verified; ready for Phase 8 disbursement eligibility. |
| `REQUEST_REVISION` | `REQUEST_REVISION` | `CONCURRING_REVISION` | `REVISION_REQUIRED` | `REVISION_REQUIRED` | Combined revision requirements sent to NGO; deadline set. |
| `REJECT` | `REJECT` | `CONCURRING_REJECT` | `REJECTED` | `REJECTED` | Adverse determination recorded; appeal window opens. |
| `APPROVE` | `REJECT` | `SPLIT_DECISION` | `ESCALATED` | `VERIFICATION_PENDING` | Sent to Platform Administrator / Senior Adjudicator for binding determination. |
| `APPROVE` | `REQUEST_REVISION` | `SPLIT_DECISION` | `ESCALATED` | `VERIFICATION_PENDING` | Sent to Platform Administrator / Senior Adjudicator for binding determination. |
| `REJECT` | `REQUEST_REVISION` | `SPLIT_DECISION` | `ESCALATED` | `VERIFICATION_PENDING` | Sent to Platform Administrator / Senior Adjudicator for binding determination. |

---

## 4. Production Lock & Immutability Rules

### Pydantic Settings Validation
In production environments (`ENVIRONMENT=production`), the dual-key mechanism cannot be disabled or weakened under any circumstances. The system enforces:
```python
if self.ENVIRONMENT == "production":
    if not self.REVIEW_SECOND_REVIEW_REQUIRED:
        raise ValueError("Dual-key approval cannot be disabled in production! REVIEW_SECOND_REVIEW_REQUIRED must be True.")
    if self.REVIEW_REQUIRED_APPROVALS < 2:
        raise ValueError("Dual-key approval cannot be disabled in production! REVIEW_REQUIRED_APPROVALS must be at least 2.")
```

### Decision Immutability
Once a review decision is recorded:
- It cannot be updated, edited, or retracted.
- Subsequent calls to `/api/v1/reviews/tasks/{task_id}/decision` return `409 Conflict`.
- If an adverse decision is successfully appealed, the original decision is marked `is_superseded = True` rather than deleted, preserving an audit trail of both the original and new outcomes.

