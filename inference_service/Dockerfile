# CUDA runtime base (includes CUDA libraries needed at runtime)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    UVICORN_WORKERS=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/torch \
    CLEARML_CACHE_DIR=/tmp/clearml

WORKDIR /app

# OS deps + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root (Cloud Run-friendly).
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

# Create writable cache directories for the non-root user
RUN mkdir -p /tmp/.cache /tmp/torch /tmp/clearml \
    && chown -R appuser:appuser /tmp/.cache /tmp/torch /tmp/clearml

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install Python deps (FastAPI/ClearML/etc.)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled PyTorch matching CUDA 12.1
# If you pin torch in requirements.txt, remove it there to avoid conflicts.
RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

COPY app.py .

EXPOSE 8080
USER appuser

HEALTHCHECK --interval=30s --timeout=2s --start-period=15s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" > /dev/null || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS} --proxy-headers --timeout-keep-alive 75 ${UVICORN_RELOAD:+--reload}"]
