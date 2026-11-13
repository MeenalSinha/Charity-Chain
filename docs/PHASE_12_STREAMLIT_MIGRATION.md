# CharityChain — Phase 12 Streamlit Migration & Coexistence Policy
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Migration Strategy Overview

Phase 12 transitions the primary user-facing interface of CharityChain from the legacy Streamlit prototype (`app_ui.py` and `charity_tracker.py`) to the production Next.js 16 App Router application.

In accordance with strict migration requirements:
- **Streamlit files (`app_ui.py`, `charity_tracker.py`) are NOT deleted.** They remain fully functional as a local fallback and reference implementation.
- All core user journeys and workflows are fully supported in Next.js.
- Next.js is configured as the default user interface (Port 3000).

---

## 2. Feature Coverage Comparison

| Capability | Streamlit (`app_ui.py`) | Next.js (`frontend/`) | Migration Status |
|---|---|---|---|
| Public Homepage & Stats | ✅ Page Home | ✅ `/` | Migrated (RSC + live API) |
| Project Catalog & Search | ✅ Public Discovery | ✅ `/projects` | Migrated (Server rendered, filters) |
| Project Details & Milestones | ✅ Project View | ✅ `/projects/[id]` | Migrated |
| NGO Public Profile | ✅ Org Explorer | ✅ `/organizations/[id]` | Migrated |
| Public Transparency Portal | ✅ Transparency Tab | ✅ `/transparency` | Migrated |
| Geographic Impact Map | ✅ Folium Leaflet | ✅ `/impact-map` | Migrated (Privacy centroids) |
| Certificate Verification | ✅ Cert Verification | ✅ `/certificates/verify/[id]` | Migrated (Ed25519 & JWKS) |
| User Sign In & Registration | ✅ Auth Tabs | ✅ `/login`, `/register` | Migrated (BFF HttpOnly cookies) |
| Donor Portfolio & Receipts | ✅ Donor Dashboard | ✅ `/dashboard/donor` | Migrated |
| Contribution Impact Journey | ✅ Journey Explorer | ✅ `/dashboard/donor` | Migrated (Visual lifecycle) |
| NGO Project Management | ✅ NGO Dashboard | ✅ `/dashboard/ngo` | Migrated |
| Reviewer Queue & Dual-Key | ✅ Reviewer Portal | ✅ `/dashboard/reviewer` | Migrated |
| Admin Ledger & Telemetry | ✅ Financial Ops | ✅ `/dashboard/admin` | Migrated (Trial balance check) |

---

## 3. Coexistence Notice in Streamlit

A transitional banner is mounted on `app_ui.py` informing users:
> **Notice:** The modern Next.js user interface is now active on port 3000 (`http://localhost:3000`). This Streamlit interface remains available for transitional reference.

