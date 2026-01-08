# ==============================
# ELEANOR V8 — Unified Engine Dockerfile
# ==============================

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./engine ./engine
COPY ./api ./api
COPY ./governance ./governance
COPY ./tests ./tests
COPY ./pyproject.toml ./
COPY ./README.md ./

# Install Python deps
RUN pip install --upgrade pip && pip install -e .

# Expose API port
EXPOSE 8000

# Default command: run REST + WebSocket server
CMD ["uvicorn", "api.rest.main:app", "--host", "0.0.0.0", "--port", "8000"]
