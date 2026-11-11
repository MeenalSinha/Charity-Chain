# CharityChain — Formal Dispute Resolution & Appeal Governance

## 1. Overview & Objective

Adverse determinations (rejections) can significantly impact NGO operations and donor trust. To prevent arbitrary, mistaken, or biased rejections, CharityChain establishes a formal, legalistic **Dispute & Appeal Workflow**.

This workflow provides NGOs with due process to contest adverse decisions, while preserving all original determinations and maintaining strict adjudicator independence.

---

## 2. Appeal Eligibility & Filing Rules

An appeal may only be filed under strict conditions:
1. **Adverse Decision Required**: Disputes can only be raised against review cases with status `REJECTED` or final decision `REJECT`.
2. **Authorized Filing**: Only verified organization administrators (`OWNER` or `ADMIN`) or platform admins may file appeals.
3. **Appeal Window Horizon**: The dispute must be submitted prior to the `appeal_deadline` (default: 168 hours / 7 days post-rejection, configured via `REVIEW_APPEAL_WINDOW_HOURS`). Submissions past the deadline return `403 Forbidden`.
4. **Appeals Ceiling**: Maximum 1 appeal per review case (`REVIEW_MAX_APPEALS_PER_CASE`), preventing endless administrative litigation.

---

## 3. Dispute Categories & Rationale

When filing a dispute via `POST /api/v1/reviews/cases/{case_id}/disputes`, the NGO must select a standardized category and provide a substantiated statement (minimum 15 characters):

| Category | Typical Justification |
|---|---|
| `EVIDENCE_MISINTERPRETATION` | Reviewer misread diagrams, misinterpreted focal lens distortion, or miscalculated physical scale. |
| `PROCEDURAL_ERROR` | Assigned reviewer failed to follow mandatory verification guidelines or omitted submitted documents. |
| `CONFLICT_OF_INTEREST` | Evidence of undisclosed reviewer bias, competitive conflict, or external malice. |
| `TECHNICAL_VERIFICATION_ISSUE` | Discrepancies caused by camera EXIF truncation, satellite cloud cover, or compression artifacts. |
| `MISSING_INFORMATION` | Reviewer lacked context regarding local seasonal conditions, legal permits, or third-party delays. |

---

## 4. Discussion Thread & Evidence Exchange

Once opened, the dispute provides a private, auditable discussion thread between the NGO leadership and the adjudicator:
- `POST /api/v1/reviews/disputes/{dispute_id}/messages` allows appending messages and documentation.
- `GET /api/v1/reviews/disputes/{dispute_id}` retrieves the complete threaded discussion.
- All messages are timestamped, attributed to verified user IDs, and recorded in the audit trail.

---

## 5. Adjudicator Independence & Binding Outcomes

Disputes are assigned to an independent Platform Administrator / Senior Adjudicator. 

### Conflict Check
The assigned adjudicator **cannot** be any of the reviewers who submitted decisions in the original review case (`adjudicator_id NOT IN prior_reviewers`). Attempting to self-adjudicate raises `403 Forbidden`.

### Binding Resolutions

| Resolution Outcome | Effect on Review Case | Effect on Milestone | Description |
|---|---|---|---|
| `OVERTURN_ORIGINAL_DECISION` | Moves to `RESOLVED` | Moves to `VERIFICATION_PENDING` | The rejection is overturned; milestone is reopened for approval or immediate completion. |
| `ORDER_RE_REVIEW` | Moves to `RESOLVED` | Moves to `VERIFICATION_PENDING` | A new review case round is generated, and a fresh panel of independent reviewers is assigned. |
| `REQUEST_NEW_EVIDENCE` | Moves to `REVISION_REQUIRED` | Moves to `REVISION_REQUIRED` | The dispute establishes that additional verification is required before a final determination. |
| `UPHOLD_ORIGINAL_DECISION` | Moves to `CLOSED` | Remains `REJECTED` | The original rejection is confirmed as valid, sound, and binding. Case is permanently closed. |
| `DISMISS_DISPUTE` | Moves to `CLOSED` | Remains `REJECTED` | Appeal is dismissed as unsubstantiated, frivolous, or vexatious. |
