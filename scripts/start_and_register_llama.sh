#!/usr/bin/env bash
set -euo pipefail

# Starts the Eleanor API locally (if not already running) and registers
# Ollama-backed critic adapters plus bindings.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="$ROOT/logs"
API_LOG="$LOG_DIR/eleanor-api.log"
API_STARTED=0

VENV_UVICORN="$ROOT/.venv/bin/uvicorn"
UVICORN_BIN="${UVICORN_BIN:-}"

if [ -z "$UVICORN_BIN" ]; then
  if [ -x "$VENV_UVICORN" ]; then
    UVICORN_BIN="$VENV_UVICORN"
  else
    UVICORN_BIN="$(command -v uvicorn || true)"
  fi
fi

start_uvicorn() {
  if [ -z "$UVICORN_BIN" ]; then
    echo "uvicorn not found. Install dependencies or set UVICORN_BIN."
    exit 1
  fi

  ELEANOR_ENV=${ELEANOR_ENV:-local} \
  CONSTITUTIONAL_CONFIG_PATH=${CONSTITUTIONAL_CONFIG_PATH:-governance/constitutional.yaml} \
  "$UVICORN_BIN" api.rest.main:app --host "$HOST" --port "$PORT" >"$API_LOG" 2>&1 &
}

start_api() {
  if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "API already running on port $PORT; skipping start."
    return
  fi

  mkdir -p "$LOG_DIR"
  echo "Starting Eleanor API (logs: $API_LOG)..."
  (cd "$ROOT" && start_uvicorn) >/dev/null
  API_STARTED=1
}

wait_for_api() {
  echo "Waiting for API on port $PORT..."
  for _ in {1..40}; do
    if curl -sf "http://localhost:$PORT/admin/router/health" >/dev/null 2>&1; then
      echo "API is ready."
      return
    fi
    sleep 2
  done

  echo "API did not become ready; check $API_LOG"
  exit 1
}

register_adapters() {
  echo "Registering Ollama adapters..."
  local models=(rights fairness risk truth pragmatics phi3)
  for name in "${models[@]}"; do
    local payload
    payload=$(printf '{"name":"ollama-%s","type":"ollama","model":"%s"}' "$name" "$name")
    curl -sf -X POST "http://localhost:$PORT/admin/router/adapters" \
      -H "Content-Type: application/json" \
      -d "$payload" >/dev/null
    echo "  adapter registered: ollama-$name"
  done
}

bind_critics() {
  echo "Binding critics..."
  local bindings=(
    "rights:ollama-rights"
    "fairness:ollama-fairness"
    "risk:ollama-risk"
    "truth:ollama-truth"
    "pragmatics:ollama-pragmatics"
    "autonomy:ollama-phi3"
  )

  for entry in "${bindings[@]}"; do
    IFS=: read -r critic adapter <<<"$entry"
    local payload
    payload=$(printf '{"critic":"%s","adapter":"%s"}' "$critic" "$adapter")
    curl -sf -X POST "http://localhost:$PORT/admin/critics/bindings" \
      -H "Content-Type: application/json" \
      -d "$payload" >/dev/null
    echo "  $critic -> $adapter"
  done
}

show_status() {
  echo "Router health:"
  curl -s "http://localhost:$PORT/admin/router/health" || true
  echo ""
  echo "Critic bindings:"
  curl -s "http://localhost:$PORT/admin/critics/bindings" || true
  echo ""
}

start_api
wait_for_api
register_adapters
bind_critics
show_status

if [ "$API_STARTED" -eq 1 ]; then
  echo ""
  echo "Eleanor API is running in the background (logs: $API_LOG)."
  echo "Press Ctrl+C to stop it or kill the process manually when done."
fi
