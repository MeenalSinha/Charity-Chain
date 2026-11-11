# Payment Security, Threat Model & PCI Scope Boundaries

## 1. Threat Model & Security Defenses

CharityChain integrates financial capabilities with a defense-in-depth security posture designed to defeat common payment and API attacks:

| Threat Vector | Attack Mechanism | CharityChain Defense |
| :--- | :--- | :--- |
| **Insecure Direct Object Reference (IDOR)** | Malicious donor attempts to capture or refund another user's order | Strict ownership validation in `DonationService`: verifies `donation.donor_id == current_user.id` or user is `PLATFORM_ADMIN`. |
| **Client-Side Amount Tampering** | Attacker intercepts checkout payload to modify donation amount or currency | All amounts and currencies are enforced server-side. Pre-capture verification compares PayPal gateway `amount.value` with database record before capture. |
| **Currency Arbitrage Manipulation** | Attacker submits low-value currency to fund high-value project target | Strict currency consistency check: raises `CurrencyMismatchException` if request currency does not equal `project.currency`. |
| **Webhook Forgery / Spoofing** | Attacker crafts fake webhook POST to trigger unauthorized captures | Mandatory cryptographic asymmetric signature verification against PayPal Public Key Certificates. |
| **Replay Attacks** | Attacker replays valid webhook or capture requests | Idempotency keys (`PayPal-Request-Id`) and deduplication table (`webhook_events.event_id UNIQUE`). |
| **Race Conditions in Refunds** | Concurrent refund requests exceeding remaining refundable balance | Database transactions and remaining refundable ceiling checks (`amount <= capture_amount - sum(refunds)`). |

---

## 2. PCI-DSS Compliance & Scope Boundary

CharityChain operates entirely within the lowest PCI-DSS compliance scope (**SAQ A**):

1. **Zero Cardholder Data Storage**: CharityChain **never** receives, handles, transmits, or stores Primary Account Numbers (PAN), Card Verification Values (CVV), or bank account numbers.
2. **Hosted Gateway Redirection**: All payment collection is offloaded to PayPal's secure checkout infrastructure (`https://www.sandbox.paypal.com` or `https://www.paypal.com`).
3. **Tokenized Referencing**: The backend interacts strictly with opaque references:
   - Provider Order ID (`ORD-...`)
   - Provider Capture ID (`CAP-...`)
   - Provider Refund ID (`REF-...`)
4. **Transport Layer Security**: All external communications with PayPal require TLS 1.2 or higher over HTTPS.

---

## 3. Administrator Auditing & Monitoring

Every critical financial event generates an immutable audit record in the `audit_events` table:
- `DONATION_PAYMENT_CAPTURED`: Logs capture ID, provider capture ID, donor ID, order ID, amount, and currency.
- `DONATION_REFUND_COMPLETED`: Logs refund ID, provider refund ID, actor ID, capture ID, amount, and reason.

Platform administrators have access to dedicated endpoints:
- `GET /api/v1/admin/payments`: Real-time transaction inspection.
- `GET /api/v1/admin/payment-reconciliation`: Discrepancy detection between local records and gateway capture states.

