# Use a slim Python 3.11 base image
FROM python:3.13-slim

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENV_MODE=prod
# Ensure uv uses the system python environment
ENV UV_SYSTEM_PYTHON=1 

# Install system-level dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the configuration files first for better caching
# We need both pyproject.toml and uv.lock (if you have one)
COPY pyproject.toml uv.lock ./

# Install CPU-specific PyTorch first (to keep image slim)
# --find-links allows uv to pick the CPU wheels
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies from pyproject.toml
RUN uv pip install . --no-cache

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/db data/uploads logs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Make start.sh executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
