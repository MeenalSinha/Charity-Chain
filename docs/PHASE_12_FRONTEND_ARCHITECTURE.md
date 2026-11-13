# CharityChain — Phase 12 Frontend Architecture
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Overview & System Topology

CharityChain's Phase 12 modernizes the user interface by establishing a production Next.js App Router web application integrating with the existing authoritative FastAPI backend.

```
                    +---------------------------+
                    |        WEB BROWSER        |
                    | (Desktop / Tablet / Mobile)
                    +-------------+-------------+
                                  |
                                  | HTTP / HTTPS (Port 3000)
                                  v
                    +---------------------------+
                    |     NEXT.JS FRONTEND      |
                    |      (Port 3000)          |
                    |                           |
                    |  * Server Components (RSC)|
                    |  * Route Handlers (BFF)   |
                    |  * HttpOnly Cookie Auth   |
                    |  * Tailwind CSS Design    |
                    +-------------+-------------+
                                  |
                                  | Server-to-Server REST
                                  | Bearer Authorization
                                  v
                    +---------------------------+
                    |      FASTAPI BACKEND      |
                    |      (Port 8000)          |
                    |                           |
                    |  * Business Logic         |
                    |  * Financial Ledger       |
                    |  * AI Verification Engine |
                    |  * Dual-Key Human Review  |
                    |  * Asymmetric Signatures  |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             +---------------+           +---------------+
             |  POSTGRESQL   |           |    REDIS 7    |
             |  Authoritative|           |   Queue &     |
             |   Database    |           |    Cache      |
             +---------------+           +---------------+
```

---

## 2. Directory Structure

```
frontend/
├── app/
│   ├── (public)/              # Unauthenticated public website
│   │   ├── layout.tsx         # PublicHeader + children + PublicFooter
│   │   ├── page.tsx           # Homepage (Hero, Stats, How it works, Featured)
│   │   ├── projects/          # Project directory & search
│   │   │   ├── page.tsx       # Filterable project catalog
│   │   │   └── [projectId]/   # Project details, milestones, donation CTA
│   │   ├── organizations/     # Verified NGO public profiles
│   │   ├── transparency/      # Public transparency portal & methodology
│   │   ├── impact-map/        # Geospatial impact map (privacy-safe)
│   │   └── certificates/      # Ed25519 proof verification
│   ├── (auth)/                # Authentication screens
│   │   ├── layout.tsx         # Centered clean auth container
│   │   ├── login/             # Login form
│   │   ├── register/          # User registration
│   │   ├── verify-email/      # Token verification
│   │   ├── forgot-password/   # Password recovery
│   │   └── reset-password/    # Password reset
│   ├── (dashboard)/           # Role-based workspace dashboards
│   │   ├── dashboard/
│   │   │   ├── donor/         # Donor portfolio, receipts, impact journeys
│   │   │   ├── ngo/           # Organization management & evidence
│   │   │   ├── reviewer/      # Review queue & dual-key signoff
│   │   │   └── admin/         # Platform telemetry & ledger reconciliation
│   ├── api/auth/              # BFF authentication route handlers
│   │   ├── login/route.ts     # Forward login & set HttpOnly cookies
│   │   ├── logout/route.ts    # Clear cookies & notify backend
│   │   ├── refresh/route.ts   # Refresh token rotation
│   │   ├── session/route.ts   # Server session introspection
│   │   ├── register/route.ts  # Forward registration
│   │   └── verify-email/      # Forward token verification
│   ├── layout.tsx             # Root layout with QueryProvider
│   └── globals.css            # Tailwind CSS 4 styling
├── components/
│   ├── ui/                    # Reusable primitive components (Badge, Button, Card, etc.)
│   ├── layout/                # Headers, Footers, Responsive Sidebar DashboardLayout
│   ├── forms/                 # React Hook Form + Zod validated forms
│   └── maps/                  # Client-side map components
├── lib/
│   ├── api/                   # Centralized typed API client & server-side helpers
│   ├── auth/                  # Session management & client auth utilities
│   ├── constants/             # Routes, pagination, currency codes
│   ├── formatters/            # Currency minor-unit, date, percent formatters
│   └── utilities/             # cn (clsx + twMerge), correlation IDs
├── types/                     # TypeScript API contracts and domain models
└── tests/                     # Unit and integration test suites (Vitest)
```

---

## 3. Core Architectural Decisions

1. **Backend as Source of Truth**: The Next.js frontend strictly renders data and handles user interactions. All business rules, monetary calculations, state transitions, and cryptographic operations execute in the FastAPI backend.
2. **BFF Security Model**: JWT tokens are securely held in HttpOnly cookies managed by Next.js Route Handlers. Client-side JavaScript cannot access raw authentication tokens.
3. **Server Components by Default**: Pages fetch data on the server with Next.js Server Components, rendering fast HTML without client-side waterfalls.
4. **Accessible Primitive Components**: All interactive components provide semantic HTML, ARIA attributes, and keyboard navigability.

