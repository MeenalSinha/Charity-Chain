# CharityChain — Notification Templates & Responsive Layout Spec

## 1. Overview

All transactional emails in CharityChain are compiled using `EmailTemplateService`. The template engine produces responsive HTML with mobile-friendly typography, emerald green branding, high-contrast call-to-action buttons, and accessible fallback typography.

---

## 2. Master Responsive Layout Architecture

Every email is encased in the master layout structure:

- **Max Width**: 600px centered table container.
- **Header**: Deep emerald background (`#064e3b`) featuring the CharityChain brandmark and motto *"Give with Evidence. Release with Confidence."*
- **Body Area**: White background (`#ffffff`), 15px font size, 1.6 line height, and distinct callout blocks with left accent borders for financial amounts and milestone titles.
- **Action Button**: Centered button (`#059669`) with 14px padding and bold white text.
- **Footer**: Neutral gray (`#f9fafb`) with regulatory notice, copyright attestation, and link to immutable impact ledger.

---

## 3. Template Inventory

### 1. `donation.captured` (Donation Receipt)
- **Subject**: `Donation Confirmed: {amount} to {project_title}`
- **Variables**: `donor_name`, `formatted_amount`, `project_title`, `milestone_title`, `order_id`, `action_url`
- **Call-to-Action**: `View My Contribution`
- **Security**: Escapes donor name and project title to prevent XSS.

### 2. `certificate.issued` (Impact Attestation)
- **Subject**: `Your Proof-of-Impact Certificate is Ready ({certificate_number})`
- **Variables**: `recipient_name`, `certificate_number`, `project_title`, `impact_summary`, `verification_url`
- **Call-to-Action**: `Verify Digital Certificate`
- **Features**: Features high-contrast certificate badge with cryptographic verification instructions.

### 3. `certificate.revoked` (Status Update)
- **Subject**: `Notice: Certificate {certificate_number} Status Changed`
- **Variables**: `certificate_number`, `reason`, `verification_url`
- **Call-to-Action**: `View Status on Ledger`
- **Features**: Prominent red alert box detailing revocation justification (e.g. full refund or governance revision).

### 4. `review.assigned` (Reviewer Task Alert)
- **Subject**: `New Review Assignment: {milestone_title}`
- **Variables**: `reviewer_name`, `milestone_title`, `project_title`, `due_date`, `action_url`
- **Call-to-Action**: `Open Reviewer Workspace`
- **Features**: Summarizes AI verification status and specifies 72-hour review deadline.

### 5. `disbursement.completed` (NGO Fund Transfer Notice)
- **Subject**: `Disbursement Transferred: {amount} for {milestone_title}`
- **Variables**: `org_name`, `formatted_amount`, `milestone_title`, `batch_id`, `action_url`
- **Call-to-Action**: `View Financial Summary`
- **Features**: Informs NGO of cleared payout and references double-entry ledger settlement record.

---

## 4. XSS Escaping & Security

All dynamic values undergo `html.escape` before injection into the HTML template:

```python
safe_donor_name = html.escape(ctx.get("donor_name", "Supporter"))
safe_project_title = html.escape(ctx.get("project_title", "Project"))
```

Attempts to inject `<script>`, `<iframe>`, or `onload` vectors are neutralized into harmless text entities (`&lt;script&gt;`).

