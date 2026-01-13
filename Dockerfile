# ==============================
# ELEANOR V8 — Unified Engine Dockerfile
# ==============================

FROM node:20-alpine AS ui-builder

WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --silent
COPY ui/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./engine ./engine
COPY ./api ./api
COPY ./governance ./governance
COPY ./precedent ./precedent
COPY ./config ./config
COPY ./pyproject.toml ./
COPY ./README.md ./

# UI assets
COPY --from=ui-builder /ui/dist ./ui/dist

# Install Python deps
RUN pip install --upgrade pip && pip install -e ".[api,vector,observability]"

# Expose API port
EXPOSE 8000

# Default command: run REST + WebSocket server
CMD ["uvicorn", "api.rest.main:app", "--host", "0.0.0.0", "--port", "8000"]
