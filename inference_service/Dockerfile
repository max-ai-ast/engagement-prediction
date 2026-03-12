FROM python:3.11-slim-bookworm

ARG GIT_SHA=""
LABEL org.opencontainers.image.revision=$GIT_SHA

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    UVICORN_WORKERS=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/torch \
    CLEARML_CACHE_DIR=/tmp/clearml \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# OS deps (keep small; Cloud Run-friendly)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root (Cloud Run-friendly).
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

# Create writable cache directories for the non-root user
RUN mkdir -p /tmp/.cache /tmp/torch /tmp/clearml \
    && chown -R appuser:appuser /tmp/.cache /tmp/torch /tmp/clearml

# Create a venv for deterministic installs
RUN python -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip

# Install Python deps (FastAPI/ClearML/etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG TORCH_VERSION=2.5.1
# CPU-only PyTorch wheel (so this image can't accidentally use CUDA even if run with --gpus).
RUN pip install --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/cpu" \
    "torch==${TORCH_VERSION}+cpu"

COPY app.py .

EXPOSE 8080
USER appuser

HEALTHCHECK --interval=30s --timeout=2s --start-period=15s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null || exit 1

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS} --proxy-headers --timeout-keep-alive 75 ${UVICORN_RELOAD:+--reload}"]
