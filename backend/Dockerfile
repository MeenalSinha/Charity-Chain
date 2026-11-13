# ============================================================
# CharityChain Backend Dockerfile — Multi-Stage Production Build
# Phase 14: Hardened non-root container, production-only deps
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix directory (production only)
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Production Runner ----
FROM python:3.11-slim AS runner

WORKDIR /app

# Install only runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN groupadd --system --gid 10001 charitychain \
    && useradd --system --uid 10001 --gid charitychain --no-create-home --shell /sbin/nologin charitychain

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source (no dev files, no test files)
COPY backend ./backend
COPY charity_tracker.py app_ui.py ./

# Copy model weights only if needed in production
# If YOLO is used at runtime, include weights; otherwise remove
COPY yolov8n.pt ./

# Create log and data directories with correct ownership
RUN mkdir -p /app/data /app/logs \
    && chown -R charitychain:charitychain /app

# Switch to non-root user
USER charitychain

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form for proper signal handling (SIGTERM from orchestrator)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
