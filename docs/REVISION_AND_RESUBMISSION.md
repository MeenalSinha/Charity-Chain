# CharityChain — Revision Requests & Resubmission Workflow

## 1. Overview & Objective

When evidence submitted by an NGO is ambiguous, incomplete, blurry, or missing key supporting documents, outright rejection is often counterproductive. CharityChain provides a structured **Revision Workflow** that allows reviewers to formally request specific clarifications and additional evidence while freezing previous submissions for complete historical accountability.

---

## 2. Structured Revision Requests

When a reviewer selects `REQUEST_REVISION`, they must provide:
1. **Detailed Rationale**: A written explanation (minimum 10-15 characters) specifying why the current evidence package is insufficient.
2. **Structured Revision Data**: A machine-readable payload specifying:
   - Specific items required (e.g. `serial_number_photo`, `invoice_pdf`).
   - Criteria for acceptance (e.g. `resolution >= 1080p`, `geotagged metadata required`).
   - Expected submission timeline.

When both reviewers concur on `REQUEST_REVISION` (or an adjudicator issues a revision order on a split case):
- Case status transitions to `REVISION_REQUIRED`.
- Milestone status transitions to `REVISION_REQUIRED`.
- `revision_deadline` is calculated based on `REVIEW_REVISION_DUE_HOURS` (default: 168 hours / 7 days).

---

## 3. NGO Resubmission Process

To respond to revision requests, an authorized NGO administrator invokes:
`POST /api/v1/reviews/cases/{case_id}/revision-response`

Payload requirements:
- `written_response`: Comprehensive explanation detailing how each reviewer request was satisfied.
- `new_evidence_file_ids`: List of UUIDs for newly uploaded replacement or supplemental evidence files.

### Multi-Round Architecture
CharityChain uses an immutable multi-round architecture:
1. **Prior Case Preservation**: The original case (Round 1) is closed (`status = CLOSED`), but its evidence links, reviewer assignments, and decision comments remain permanently preserved in the database.
2. **New Case Creation (Round N+1)**: A new `ReviewCase` is created with:
   - `submission_round = prior_round + 1`
   - New frozen `snapshot_digest` binding the new evidence files.
   - Reference to the previous case in audit events.
3. **Milestone Reopening**: Milestone status updates to `VERIFICATION_PENDING`.
4. **Independent Re-Review**: Fresh reviewers can be assigned to evaluate Round 2, with full access to historical Round 1 findings and the NGO's written justification.

---

## 4. Deadline Enforcement

If the `revision_deadline` expires before the NGO submits a response:
- Resubmissions via the API are blocked (`403 Forbidden`).
- The case is flagged for administrative review, where the platform administrator may either extend the deadline with documented justification or transition the case to `REJECTED`.

