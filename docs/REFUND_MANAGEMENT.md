# Refund Management & Financial Adjustment Architecture

## 1. Refund Policy & Lifecycle

CharityChain provides high-assurance refund management allowing donors and authorized platform administrators to reverse charitable contributions either partially or in full.

### Core Business Rules:
1. **Refund Eligibility**: Only donations in `PAID` or `PARTIALLY_REFUNDED` status can be refunded. Donations that are `CREATED`, `PAYMENT_PENDING`, `CANCELLED`, or already fully `REFUNDED` cannot be refunded.
2. **Remaining Refundable Ceiling**:
   $$\text{Remaining Refundable Balance} = \text{Captured Amount} - \sum \text{Completed Refunds}$$
   The requested refund amount must be strictly greater than 0 and less than or equal to the remaining refundable balance.
3. **Authorization**:
   - Standard donors can **only** request refunds on donations that they personally created (`donation.donor_id == current_user.id`).
   - Platform Administrators (`PlatformRole.PLATFORM_ADMIN`) can execute refunds on any donation on behalf of users or in dispute scenarios.
4. **Gateway Execution**: No refund record is finalized locally without a confirmed `COMPLETED` response from the PayPal Payments v2 Refund API (`POST /v2/payments/captures/{id}/refund`).

---

## 2. Partial vs. Full Refunds

### Partial Refunds
- A donor who contributed \$100.00 can request a partial refund of \$40.00 (e.g. if they made an input error).
- `PaymentRefund` record is persisted with status `COMPLETED` and provider refund ID (`REF-...`).
- `Donation.status` transitions from `PAID` $\rightarrow$ `PARTIALLY_REFUNDED`.
- Net raised amounts on the associated Project and Milestone are decremented by \$40.00.
- The remaining refundable ceiling becomes \$60.00.

### Full Refunds
- If the remaining \$60.00 is subsequently refunded:
- `Donation.status` transitions from `PARTIALLY_REFUNDED` $\rightarrow$ `REFUNDED`.
- Net raised amounts on the associated Project and Milestone are decremented by the remaining \$60.00 (reaching \$0.00).
- The remaining refundable ceiling becomes \$0.00.
- Any subsequent refund attempt is immediately rejected with `PaymentRefundFailedException`.

---

## 3. Real-Time Net Funding Adjustment

Net funding is calculated as:
$$\text{Net Received Funding} = \text{Gross Captures} - \text{Completed Refunds}$$

When a refund is finalized:
1. `milestone.raised_amount_minor_units` is decremented by `refund.amount_minor_units` (floored at 0).
2. `project.raised_amount_minor_units` is decremented by `refund.amount_minor_units` (floored at 0).
3. The platform audit log records an immutable `AuditEvent` (`action = "DONATION_REFUND_COMPLETED"`).
4. Public project displays and funding progress bars immediately reflect the updated net amount.

---

## 4. REST API Refund Endpoints

- `POST /api/v1/donations/{id}/refunds`: Request full or partial refund (authenticated donor or platform admin).
- `GET /api/v1/refunds/{id}`: Inspect details of a specific refund.
- `GET /api/v1/admin/refunds`: Platform admin query listing all refunds across projects with filtering and pagination.
