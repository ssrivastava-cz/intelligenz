# syntax=docker/dockerfile:1

# =============================================================================
# Release Team Intelligenz — FastAPI backend
#
# Multi-stage:
#   builder  -> resolve + install Python deps into an isolated venv
#   runtime  -> slim image with just the venv, the app package, and libgomp1
#
# Build context is the backend/ directory:
#   docker build -t intelligenz-backend ./backend
#
# The app resolves every persistence path (data/chroma, data/bm25,
# data/history, data/redmine, database/, uploads/, logs/) and source_of_truth/
# RELATIVE TO THE WORKDIR below and creates the leaf directories lazily, so
# only the parents need to exist and be writable. On EC2 they become Docker
# volumes / bind mounts (see the project's Docker notes); the directories
# created here are the mountpoints and the fallback when nothing is mounted.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# build-essential is a compile fallback for C/C++ extensions (chroma-hnswlib,
# mmh3) on an arch/version without a prebuilt wheel — e.g. arm64 Graviton.
# It never reaches the final image.
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime
# libgomp1: OpenMP runtime that onnxruntime (pulled in transitively by
# chromadb) needs — without it, importing chromadb fails on debian-slim with
# "libgomp.so.1: cannot open shared object file".
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with a fixed UID/GID so host bind-mount ownership is
# predictable (chown the host dir to 1000:1000, or run with --user).
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --home-dir /app --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application package only — no .env, no tests, no local data (see .dockerignore).
COPY --chown=1000:1000 app/ ./app/

# Runtime working directories / mountpoints, owned by the non-root user.
RUN mkdir -p /app/data /app/database /app/uploads /app/logs /app/source_of_truth \
    && chown -R 1000:1000 /app

USER app

EXPOSE 8000

# Follows API_PREFIX (default /api/v1). start-period covers first-boot ChromaDB
# import plus the BM25 index load in the FastAPI lifespan hook.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"]

# Single worker on purpose: ChromaDB's embedded PersistentClient and the
# in-process persistent BM25 index are per-process — multiple uvicorn workers
# would hold divergent in-memory BM25 state after an index run. Scale out with
# more containers behind a load balancer, not with --workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
