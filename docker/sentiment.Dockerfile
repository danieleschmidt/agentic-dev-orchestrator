# Multi-stage Dockerfile for Sentiment Analysis Service
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build environment
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[llm,monitoring,performance] && \
    pip install --no-cache-dir pytest pytest-cov pytest-asyncio

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY ado.py ./
COPY pyproject.toml setup.py ./

# Create necessary directories
RUN mkdir -p .ado/cache logs backlog && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Environment variables
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_CACHE_DIR="/app/.ado/cache" \
    LOG_LEVEL="INFO"

# Default command
CMD ["python", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "5000"]