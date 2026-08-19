# syntax=docker/dockerfile:1

# ---- Stage 1: build a clean virtualenv with only serving dependencies ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch keeps the image small; override the index for a GPU base image.
COPY requirements-serve.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-serve.txt

# ---- Stage 2: slim runtime ----
FROM python:3.12-slim AS runtime

# Non-root user
RUN useradd --create-home --uid 1000 appuser
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    SLM_LOG_JSON=true

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY src/ ./src/

USER appuser
EXPOSE 8000

# Liveness probe hits the health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
