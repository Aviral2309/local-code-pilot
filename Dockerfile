# AuraLSP Dockerfile
# ------------------
# Builds a containerized version of the AuraLSP LSP server.
# Note: Ollama runs as a SEPARATE container (see docker-compose.yml).
#
# Build: docker build -t auralsp .
# Run:   docker-compose up

FROM python:3.11-slim

# System dependencies for Tree-Sitter compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first (layer caching: only re-run pip if deps changed)
COPY pyproject.toml ./

# Install dependencies (excluding optional dev deps)
RUN pip install --no-cache-dir -e "."

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash auralsp
RUN chown -R auralsp:auralsp /app
USER auralsp

# LSP communicates via stdin/stdout — no port needed
# The metrics API runs separately on port 8765
EXPOSE 8765

# Health check: verify Python environment is intact
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from src.config import load_config; load_config()" || exit 1

# Default: run the metrics API
# Override with: docker run auralsp python -m src.server (for LSP mode)
CMD ["python", "-m", "src.metrics_api"]
