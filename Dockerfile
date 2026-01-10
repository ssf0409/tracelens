# Agent Eval Framework - Test Environment
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first (for better layer caching)
COPY pyproject.toml README.md ./

# Copy source code (needed for editable install)
COPY src/ src/

# Install dependencies and package
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Copy tests (after install for better caching)
COPY tests/ tests/

# Default command: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]
