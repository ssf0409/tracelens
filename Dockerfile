# TraceLens - Test Environment
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Set working directory
WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first (for better layer caching)
COPY pyproject.toml README.md ./

# Copy source code (needed for editable install)
COPY src/ src/

# Install dependencies and package using uv
RUN uv pip install -e ".[dev]"

# Copy tests (after install for better caching)
COPY tests/ tests/

# Default command: run tests
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]
