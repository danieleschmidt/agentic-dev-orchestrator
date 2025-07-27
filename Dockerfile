# Multi-stage Dockerfile for Agentic Development Orchestrator
# Optimized for production deployment with security and performance

# =============================================================================
# Stage 1: Base Python Image
# =============================================================================
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    curl \
    wget \
    git \
    # Build dependencies
    build-essential \
    gcc \
    # Networking
    ca-certificates \
    # Security
    gnupg \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ado && useradd -r -g ado -d /app -s /bin/bash ado

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 2: Dependencies Installation
# =============================================================================
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

# =============================================================================
# Stage 3: Development Image
# =============================================================================
FROM dependencies AS development

# Install additional development tools
RUN pip install \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    pre-commit \
    jupyter \
    ipykernel

# Copy source code
COPY --chown=ado:ado . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/docs/status /app/escalations /app/.ado/cache /app/.ado/locks \
    && chown -R ado:ado /app

# Switch to non-root user
USER ado

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ado; print('Health check passed')" || exit 1

# Default command for development
CMD ["python", "ado.py", "help"]

# =============================================================================
# Stage 4: Production Build
# =============================================================================
FROM base AS builder

# Copy requirements and install production dependencies only
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Build the package
RUN pip install build && \
    python -m build --wheel && \
    pip install dist/*.whl

# =============================================================================
# Stage 5: Production Image (Final)
# =============================================================================
FROM python:3.11-slim-bullseye AS production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHON_ENV=production \
    ADO_LOG_LEVEL=INFO

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r ado && useradd -r -g ado -d /app -s /bin/bash ado

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm -f /tmp/*.whl

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/backlog \
    /app/docs/status \
    /app/escalations \
    /app/.ado/cache \
    /app/.ado/locks \
    /app/.ado/logs \
    && chown -R ado:ado /app

# Copy configuration files
COPY --chown=ado:ado .env.example /app/.env.example
COPY --chown=ado:ado backlog.yml /app/backlog.yml 2>/dev/null || true

# Switch to non-root user
USER ado

# Health check for production
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD ado status > /dev/null 2>&1 || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["ado", "run"]

# =============================================================================
# Stage 6: Minimal Runtime (Ultra-slim)
# =============================================================================
FROM python:3.11-alpine AS minimal

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_ENV=production

# Install minimal system dependencies
RUN apk add --no-cache \
    git \
    curl \
    ca-certificates

# Create user
RUN addgroup -g 1000 ado && \
    adduser -D -s /bin/sh -u 1000 -G ado ado

# Set working directory
WORKDIR /app

# Copy built wheel and install
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -f /tmp/*.whl

# Create directories
RUN mkdir -p \
    /app/backlog \
    /app/docs/status \
    /app/escalations \
    /app/.ado \
    && chown -R ado:ado /app

# Switch to non-root user
USER ado

# Simple health check
HEALTHCHECK --interval=60s --timeout=10s --retries=2 \
    CMD ado --help > /dev/null || exit 1

# Default command
CMD ["ado", "run"]

# =============================================================================
# Stage 7: Testing Image
# =============================================================================
FROM dependencies AS testing

# Install testing dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    tox \
    coverage

# Copy source code
COPY --chown=ado:ado . .

# Install package in development mode
RUN pip install -e .

# Create test directories
RUN mkdir -p /app/test-results /app/coverage \
    && chown -R ado:ado /app

# Switch to non-root user
USER ado

# Default command for testing
CMD ["pytest", "-v", "--cov=.", "--cov-report=html:/app/coverage", "--junit-xml=/app/test-results/junit.xml"]

# =============================================================================
# Stage 8: Documentation Builder
# =============================================================================
FROM base AS docs

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    mkdocs \
    mkdocs-material

# Copy source and docs
COPY . .

# Build documentation
RUN sphinx-build -b html docs docs/_build/html && \
    mkdocs build

# Expose documentation port
EXPOSE 8000

# Serve documentation
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]

# =============================================================================
# Metadata and Labels
# =============================================================================

# Apply labels to all stages
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0

LABEL org.opencontainers.image.title="Agentic Development Orchestrator" \
      org.opencontainers.image.description="A CLI and GitHub Action that unifies multi-agent development workflows" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.url="https://github.com/terragon-labs/agentic-dev-orchestrator" \
      org.opencontainers.image.source="https://github.com/terragon-labs/agentic-dev-orchestrator" \
      org.opencontainers.image.documentation="https://github.com/terragon-labs/agentic-dev-orchestrator/blob/main/README.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      maintainer="Terragon Labs <noreply@terragonlabs.com>"

# =============================================================================
# Build Examples:
# =============================================================================
# 
# Development:
# docker build --target development -t ado:dev .
# 
# Production:
# docker build --target production -t ado:latest .
# 
# Minimal:
# docker build --target minimal -t ado:minimal .
# 
# Testing:
# docker build --target testing -t ado:test .
# 
# Documentation:
# docker build --target docs -t ado:docs .
# 
# With build args:
# docker build \
#   --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
#   --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#   --build-arg VERSION=0.1.0 \
#   --target production \
#   -t ado:0.1.0 .
# =============================================================================