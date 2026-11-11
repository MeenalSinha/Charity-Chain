# Charitable Giving & Donation Workflow Architecture

## 1. End-to-End Donation Lifecycle

CharityChain enforces a transparent, multi-stage donation process connecting donor intentions to verified project milestones.

```
[Donor Discovery] 
       │
       ▼ Select Project & Milestone
[Create Donation Intent] ────────► Validates:
       │                            - Project is ACTIVE & PUBLIC
       │                            - NGO is VERIFIED
       │                            - Milestone is ACTIVE
       │                            - Currency matches Project
       │                            - Amount in bounds ($1.00 - $100,000.00)
       ▼
[PayPal Order Created] ──────────► Donation record (PAYMENT_PENDING)
       │                            PaymentOrder record (CREATED)
       │                            Returns Approval URL
       ▼
[Donor Approves at PayPal] 
       │
       ├─────────────────────────────────┐
       │ (Browser Return Flow)           │ (Asynchronous Webhook Flow)
       ▼                                 ▼
[Capture Endpoint Called]        [CHECKOUT.ORDER.APPROVED Received]
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
            [Backend Verify & Capture]
                      │
                      ▼
            [Payment Settled (PAID)]
                      │
                      ├─► PaymentCapture created
                      ├─► Milestone raised_amount updated
                      ├─► Project raised_amount updated
                      └─► AuditEvent logged
```

---

## 2. Server-Side Eligibility Validation Rules

Before contacting the payment gateway or creating local records, `DonationService.create_donation_intent()` executes a comprehensive sequence of non-bypassable validation checks:

| Validation Rule | Required Condition | Error Raised if Violated |
| :--- | :--- | :--- |
| **Supported Currency** | Currency in `['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'CHF', 'JPY', 'INR']` | `ValidationException` (HTTP 400) |
| **Minimum Amount** | Amount $\ge$ `MIN_DONATION_AMOUNT` (100 minor units / $1.00) | `InvalidPaymentAmountException` (HTTP 400) |
| **Maximum Amount** | Amount $\le$ `MAX_DONATION_AMOUNT` (10,000,000 minor units / $100,000.00) | `InvalidPaymentAmountException` (HTTP 400) |
| **Project Existence** | Project ID exists in database | `NotFoundException` (HTTP 404) |
| **Project Archive Status** | `project.is_archived == False` | `ValidationException` (HTTP 400) |
| **Project Status** | `project.status == ProjectStatus.ACTIVE` | `ValidationException` (HTTP 400) |
| **Project Visibility** | `project.public_visibility == ProjectVisibility.PUBLIC` | `ValidationException` (HTTP 400) |
| **Sponsoring NGO** | `organization.status == OrganizationStatus.VERIFIED` | `ValidationException` (HTTP 400) |
| **Milestone Existence** | Milestone ID exists in database | `NotFoundException` (HTTP 404) |
| **Milestone Hierarchy** | `milestone.project_id == project.id` | `ValidationException` (HTTP 400) |
| **Milestone Status** | `milestone.status` not in `(CANCELLED, COMPLETED)` | `ValidationException` (HTTP 400) |
| **Currency Consistency** | `donation.currency == project.currency` | `CurrencyMismatchException` (HTTP 422) |

---

## 3. Financial State Machine Transitions

Donations, Payment Orders, and Payment Captures operate under deterministic state machines validated by `StateMachineValidator`:

### A. Donation Status Transitions
- `CREATED` $\rightarrow$ `PAYMENT_PENDING` (PayPal order registered)
- `PAYMENT_PENDING` $\rightarrow$ `PAID` (Payment captured successfully)
- `PAYMENT_PENDING` $\rightarrow$ `FAILED` (Gateway decline or timeout)
- `PAYMENT_PENDING` $\rightarrow$ `CANCELLED` (Donor cancelled before checkout)
- `PAID` $\rightarrow$ `PARTIALLY_REFUNDED` (Cumulative refund $<$ captured amount)
- `PAID` $\rightarrow$ `REFUNDED` (Cumulative refund $==$ captured amount)
- `PARTIALLY_REFUNDED` $\rightarrow$ `REFUNDED` (Remaining balance fully refunded)

### B. Payment Order Status Transitions
- `CREATED` $\rightarrow$ `APPROVED` (Donor approved checkout)
- `CREATED` $\rightarrow$ `COMPLETED` (Direct capture)
- `CREATED` $\rightarrow$ `VOIDED` (Donor cancellation)
- `APPROVED` $\rightarrow$ `COMPLETED` (Capture settlement)
- `APPROVED` $\rightarrow$ `VOIDED` (Order expired or rejected)

### C. Payment Capture Status Transitions
- `PENDING` $\rightarrow$ `COMPLETED` (Settled)
- `PENDING` $\rightarrow$ `DECLINED` (Rejected)
- `COMPLETED` $\rightarrow$ `PARTIALLY_REFUNDED`
- `COMPLETED` $\rightarrow$ `REFUNDED`
- `PARTIALLY_REFUNDED` $\rightarrow$ `REFUNDED`

---

## 4. Dual-Path Reconciliation (Browser Return & Asynchronous Webhook)

A recurring failure mode in web payments occurs when donors complete their payment at PayPal but close the browser window before being redirected to the merchant's return page.

CharityChain eliminates this vulnerability through dual-path reconciliation:
1. **Primary Browser Return**: If the donor returns normally, `POST /api/v1/payments/orders/{order_identifier}/capture` executes and marks the donation `PAID`.
2. **Fallback Asynchronous Webhook**: If the donor closes the tab, PayPal delivers a `CHECKOUT.ORDER.APPROVED` webhook to `/api/v1/webhooks/paypal`. The webhook processor automatically invokes `capture_donation_payment()`, capturing the funds and updating project balances without donor browser interaction.
3. **Idempotency Guarantee**: If both the browser return and the webhook fire concurrently, the second execution detects that the order is already captured, avoids duplicate gateway capture requests, and returns the existing capture record.

