# CharityChain — Phase 12 Deployment & Operation Guide
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Local Development Startup

### 1.1 Prerequisites
- Python 3.12+ (or 3.14)
- Node.js 20+ (Node v22 installed)
- pnpm 10+ (or 11+)
- Docker & Docker Compose (optional for containerized setup)

### 1.2 Starting the FastAPI Backend
From repository root:
```bash
# In Terminal 1: Run FastAPI backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### 1.3 Starting the Next.js Frontend
From repository root:
```bash
# In Terminal 2: Run Next.js frontend
cd frontend
pnpm dev
```
Web application will be accessible at: `http://localhost:3000`

---

## 2. Docker Compose Deployment

The whole stack can be started with Docker Compose:
```bash
docker-compose up --build
```

### Services Started:
- **`postgres`**: Port 5432 (PostgreSQL 16)
- **`redis`**: Port 6379 (Redis 7)
- **`backend`**: Port 8000 (FastAPI API server)
- **`frontend`**: Port 3000 (Next.js production web server)

---

## 3. Environment Variables Reference

### Frontend Configuration (`frontend/.env.local` or `.env.example`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_APP_URL=http://localhost:3000
COOKIE_SECURE=false
COOKIE_SAME_SITE=lax
NEXT_PUBLIC_DEFAULT_MAP_LAT=20
NEXT_PUBLIC_DEFAULT_MAP_LNG=0
NEXT_PUBLIC_DEFAULT_MAP_ZOOM=2
```

