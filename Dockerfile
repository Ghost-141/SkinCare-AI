# --- Build Stage ---
FROM python:3.13-slim AS builder

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better caching
COPY pyproject.toml uv.lock ./
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install . --no-cache

# --- Final Production Stage ---
FROM python:3.13-slim

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV_MODE=prod \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/tmp/matplotlib

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r skincare && useradd -r -g skincare skincare

WORKDIR /app

# Copy installed packages from the builder stage
# (Assuming UV_SYSTEM_PYTHON=1 installs to standard site-packages)
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p data/db data/uploads logs && \
    chown -R skincare:skincare /app && \
    chmod -R 775 /app/data /app/logs

# Switch to non-root user
USER skincare

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Add healthcheck for ECS/Fargate
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Ensure start.sh is executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
