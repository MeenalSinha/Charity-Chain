# CharityChain — Phase 12 Frontend API Inventory
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Authentication & Identity APIs

| Frontend Feature | API Endpoint | HTTP Method | Auth Required | Required Role | Request Schema | Response Schema | Backend Change? |
|---|---|---|---|---|---|---|---|
| User Sign In | `/auth/login` | POST | No | None | `LoginRequest` | `StandardResponse[TokenResponse]` | No |
| User Registration | `/auth/register` | POST | No | None | `RegisterRequest` | `StandardResponse[RegisterResponse]` | No |
| Email Verification | `/auth/verify-email` | POST | No | None | `VerifyEmailRequest` | `StandardResponse[dict]` | No |
| Password Recovery | `/auth/forgot-password` | POST | No | None | `ForgotPasswordRequest` | `StandardResponse[dict]` | No |
| Password Reset | `/auth/reset-password` | POST | No | None | `ResetPasswordRequest` | `StandardResponse[dict]` | No |
| Token Rotation | `/auth/refresh` | POST | No | None | `RefreshTokenRequest` | `StandardResponse[TokenResponse]` | No |
| User Sign Out | `/auth/logout` | POST | Yes | Any | None | `StandardResponse[dict]` | No |
| User Profile | `/users/me` | GET | Yes | Any | None | `StandardResponse[UserProfile]` | No |

---

## 2. Public Transparency & Discovery APIs

| Frontend Feature | API Endpoint | HTTP Method | Auth Required | Required Role | Request Schema | Response Schema | Backend Change? |
|---|---|---|---|---|---|---|---|
| Global Platform Overview | `/transparency/overview` | GET | No | None | None | `StandardResponse[PublicOverview]` | No |
| Public Project Catalog | `/transparency/projects` | GET | No | None | Query filters | `StandardResponse[List[PublicProjectItem]]` | No |
| Project Transparency Detail | `/transparency/projects/{id}` | GET | No | None | Path ID | `StandardResponse[PublicProjectDetail]` | No |
| Public NGO Profile | `/transparency/organizations/{id}` | GET | No | None | Path ID | `StandardResponse[PublicOrgProfile]` | No |
| Impact Map Markers | `/transparency/map/markers` | GET | No | None | None | `StandardResponse[List[MapMarker]]` | No |
| Methodology Statement | `/transparency/methodology` | GET | No | None | None | `StandardResponse[Methodology]` | No |
| Certificate Verification | `/certificates/verify/{id}` | GET | No | None | Path ID | `StandardResponse[VerificationData]` | No |

---

## 3. Donor Portfolio APIs

| Frontend Feature | API Endpoint | HTTP Method | Auth Required | Required Role | Request Schema | Response Schema | Backend Change? |
|---|---|---|---|---|---|---|---|
| Donor Overview & Giving KPIs | `/analytics/donor/overview` | GET | Yes | DONOR | None | `StandardResponse[DonorOverview]` | No |
| Donation Portfolio & Receipts | `/analytics/donor/portfolio` | GET | Yes | DONOR | None | `StandardResponse[DonorPortfolio]` | No |
| Contribution Impact Journey | `/analytics/donor/journey` | GET | Yes | DONOR | None | `StandardResponse[List[JourneyItem]]` | No |
| Tax Statement Report | `/analytics/donor/annual-report` | GET | Yes | DONOR | Year query | `StandardResponse[AnnualReport]` | No |

---

## 4. NGO Operations APIs

| Frontend Feature | API Endpoint | HTTP Method | Auth Required | Required Role | Request Schema | Response Schema | Backend Change? |
|---|---|---|---|---|---|---|---|
| NGO Overview | `/analytics/ngo/{id}/overview` | GET | Yes | NGO Member | Org ID | `StandardResponse[NGOOverview]` | No |
| NGO Funding Breakdown | `/analytics/ngo/{id}/funding` | GET | Yes | NGO Member | Org ID | `StandardResponse[NGOFunding]` | No |
| Project Management | `/projects` | GET, POST | Yes | NGO Member | Project create schema | `StandardResponse[Project]` | No |
| Milestone Management | `/milestones/{id}` | GET, PATCH | Yes | NGO Member | Milestone schema | `StandardResponse[Milestone]` | No |
| Evidence File Ingestion | `/milestones/{id}/evidence` | POST | Yes | NGO Member | Multipart Form | `StandardResponse[EvidenceBatch]` | No |

---

## 5. Reviewer & Administrator APIs

| Frontend Feature | API Endpoint | HTTP Method | Auth Required | Required Role | Request Schema | Response Schema | Backend Change? |
|---|---|---|---|---|---|---|---|
| Review Cases Queue | `/reviews/cases` | GET | Yes | REVIEWER | Limit/offset | `StandardResponse[List[Case]]` | No |
| Decision Submission | `/reviews/cases/{id}/decision` | POST | Yes | REVIEWER | Decision payload | `StandardResponse[DecisionRecord]` | No |
| Admin Platform Telemetry | `/analytics/admin/overview` | GET | Yes | PLATFORM_ADMIN | None | `StandardResponse[AdminOverview]` | No |
| Financial & Ledger Health | `/analytics/admin/financial` | GET | Yes | PLATFORM_ADMIN | None | `StandardResponse[FinancialHealth]` | No |
| System Operational Telemetry | `/analytics/admin/health` | GET | Yes | PLATFORM_ADMIN | None | `StandardResponse[OperationalHealth]` | No |

