# CharityChain — Phase 12 Application Route Map
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Route Hierarchy

| Route Path | Type | Access Level | Description |
|---|---|---|---|
| `/` | Page (RSC) | Public | Main landing page: hero, platform statistics, how it works, featured initiatives |
| `/projects` | Page (RSC) | Public | Project catalog with category filters and search |
| `/projects/[projectId]` | Page (RSC) | Public | Project detail view: milestones, progress bar, evidence status, donation CTA |
| `/organizations/[organizationId]` | Page (RSC) | Public | Verified NGO profile with mission statement and published projects |
| `/transparency` | Page (RSC) | Public | Public transparency dashboard: global KPIs and calculation methodology |
| `/impact-map` | Page (Client) | Public | Privacy-safe geographic impact map with project markers |
| `/certificates/verify/[certificateId]` | Page (RSC) | Public | Cryptographic Ed25519 proof verification and certificate inspection |
| `/login` | Page (Client) | Guest | User authentication form with anti-enumeration protection |
| `/register` | Page (Client) | Guest | User registration with password strength requirements |
| `/verify-email` | Page (Client) | Guest | Email verification callback with token verification |
| `/forgot-password` | Page (Client) | Guest | Password reset request form |
| `/reset-password` | Page (Client) | Guest | New password submission form |
| `/dashboard/donor` | Page (RSC) | DONOR | Donor overview: contributions, impact journeys, certificates, donation receipts |
| `/dashboard/ngo` | Page (RSC) | NGO Member | NGO operations: project creation, milestone status, evidence uploads |
| `/dashboard/reviewer` | Page (RSC) | REVIEWER | Review queue: evidence viewer, AI forensics, dual-key decision actions |
| `/dashboard/admin` | Page (RSC) | PLATFORM_ADMIN | Platform telemetry: double-entry ledger trial balance, worker health, statistics |

---

## 2. API Route Handlers (BFF)

| Route Path | Method | Purpose |
|---|---|---|
| `/api/auth/login` | POST | Authenticates credentials with FastAPI, sets `cc_access_token` and `cc_refresh_token` cookies |
| `/api/auth/logout` | POST | Invalidates session on FastAPI, removes cookies from browser |
| `/api/auth/refresh` | POST | Performs refresh token rotation, updates cookies |
| `/api/auth/session` | GET | Returns session state and short-lived client access token |
| `/api/auth/register` | POST | Proxies registration request to FastAPI |
| `/api/auth/verify-email` | POST | Proxies email verification token to FastAPI |
| `/api/auth/forgot-password` | POST | Proxies password recovery request to FastAPI |
| `/api/auth/reset-password` | POST | Proxies password reset request to FastAPI |

