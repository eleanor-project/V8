#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm (Node 18+) is required"; exit 1; }

# Local defaults
export ELEANOR_ENV=${ELEANOR_ENV:-development}
export AUTH_ENABLED=${AUTH_ENABLED:-false}
export JWT_SECRET=${JWT_SECRET:-dev-secret}
export ELEANOR_CONFIG=${ELEANOR_CONFIG:-./config/eleanor.yaml}

echo "Preparing Python env..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/python -m ensurepip --upgrade >/dev/null
fi
source .venv/bin/activate
# Install deps only if core packages missing; avoid network by disabling build isolation.
if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  PIP_NO_BUILD_ISOLATION=1 pip install -q -e .
fi

echo "Building UI..."
cd ui
npm install --silent
npm run build
cd "$ROOT"

echo "Starting API + UI at http://localhost:8000/ui/ ..."
if command -v open >/dev/null 2>&1; then
  (sleep 2 && open "http://localhost:8000/ui/") >/dev/null 2>&1 &
fi
uvicorn api.rest.main:app --host 0.0.0.0 --port 8000
