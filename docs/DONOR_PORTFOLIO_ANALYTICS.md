# CharityChain — Donor Portfolio Analytics & Impact Journey

## 1. Overview & Purpose
Phase 11 introduces a comprehensive, multi-currency donor intelligence platform. Instead of receiving generic receipts, donors can view an interactive portfolio that tracks every dollar contributed from the moment of payment capture through to physical milestone delivery, AI forensic verification, human reviewer sign-off, and cryptographic Proof-of-Impact certification.

---

## 2. Multi-Currency Segregation
CharityChain operates globally and accepts contributions in multiple currencies (USD, EUR, GBP, CAD, AUD, etc.).
- **Invariant**: Donations in different currencies are **never** added together into a single arbitrary scalar without explicit currency identification.
- **Bucketing**: Each currency maintains independent totals for:
  - `captured_cents`: Total successful donations in minor units.
  - `refunded_cents`: Total refunded amounts in minor units.
  - `net_cents`: Net contributed minor units.
- **Display Formats**: Formatted with standard international currency symbols (`$100.00`, `€50.00`).

---

## 3. End-to-End Personal Impact Journey
Every contribution is mapped through its full lifecycle via `DonorAnalyticsService.get_personal_impact_journey`:
```
Donation (Payment Captured)
  │
  ▼
Project & Milestone Allocation
  │
  ▼
NGO Evidence Submission (Photos, Invoices, GPS Coordinates)
  │
  ▼
AI Evidence Verification (ELA, pHash, YOLO Object Detection, Metadata Forensics)
  │
  ▼
Dual-Key Human Review (Independent Peer Review & Conflict Checking)
  │
  ▼
Disbursement & Cryptographic Proof-of-Impact Certificate (Ed25519 Signed)
```

### Attributes Tracked per Contribution
- **Donation ID & Timestamp**: Unique transaction reference and ISO-8601 creation time.
- **Net Contribution Amount**: Integer minor units and formatted currency string.
- **Target Project & Organization**: Project title, public visibility, and verified NGO name.
- **Milestone Status**: Current state (`ACTIVE`, `EVIDENCE_SUBMITTED`, `VERIFICATION_PENDING`, `COMPLETED`).
- **Evidence Presence**: Total evidence files attached to the milestone.
- **Certificate Issuance**: Certificate number, verification ID, and public verification link.

---

## 4. Annual Giving Statement (Tax Disclosure)
Donors can generate an official annual giving report covering a given calendar year:
- **Eligible Items**: All donations with `status == DonationStatus.PAID` created between Jan 1 00:00:00 UTC and Dec 31 23:59:59 UTC of the target year.
- **Summary**: Total contributions, itemized list of supported charities, dates, and amounts.
- **Mandatory Tax Disclosure Disclaimer**:
  > *"This statement is provided for informational purposes based on recorded payments on the CharityChain platform. CharityChain is a technology platform and does not provide formal legal or tax advice. Donors should verify the tax-deductible status of recipient organizations with qualified tax professionals according to their local tax jurisdiction."*

---

## 5. Security & Privacy Guarantees
- **Strict Donor Ownership**: Donors can only access their own portfolio and tax reports (`/api/v1/analytics/donor/overview`, `/api/v1/analytics/donor/journey`, etc.). Platform admins have read-only audit access.
- **Anonymous Giving**: If a donor chooses `is_anonymous = True`, their identity is permanently stripped from all public transparency feeds and NGO public reports.
- **Zero Information Leakage**: Other donors cannot query or enumerate another user's contributions.
