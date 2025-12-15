#!/usr/bin/env bash

# Start the Eleanor API and register local Ollama critic adapters/bindings.
# The script will:
#   1) Ensure uvicorn is available (prefers .venv, creates it if needed)
#   2) Launch the API (or reuse an already running instance)
#   3) Register Ollama adapters for each critic
#   4) Bind critics to those adapters
#   5) Print router health and bindings

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_URL="${API_URL:-http://${HOST}:${PORT}}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
LOG_DIR="$ROOT/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eleanor-api.log}"

# Default models (override to match what you have pulled in Ollama)
CRITIC_MODEL="${CRITIC_MODEL:-llama3}"
AUTONOMY_MODEL="${AUTONOMY_MODEL:-phi3}"

# API configuration
export ELEANOR_ENV="${ELEANOR_ENV:-local}"
export CONSTITUTIONAL_CONFIG_PATH="${CONSTITUTIONAL_CONFIG_PATH:-governance/constitutional.yaml}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

UVICORN_BIN=""
API_PID=""
STARTED_API=0

ensure_uvicorn() {
  if [ -x "$ROOT/.venv/bin/uvicorn" ]; then
    UVICORN_BIN="$ROOT/.venv/bin/uvicorn"
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    return
  fi

  if command -v uvicorn >/dev/null 2>&1; then
    UVICORN_BIN="$(command -v uvicorn)"
    return
  fi

  echo "Setting up Python venv and installing dependencies..."
  python3 -m venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  .venv/bin/python -m ensurepip --upgrade >/dev/null
  PIP_NO_BUILD_ISOLATION=1 pip install -q -e .
  UVICORN_BIN="$ROOT/.venv/bin/uvicorn"
}

api_running() {
  curl -fsS --max-time 2 "$API_URL/health" >/dev/null 2>&1
}

start_api() {
  echo "Starting Eleanor API (logs: $LOG_FILE)..."
  "$UVICORN_BIN" api.rest.main:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
  API_PID=$!
  STARTED_API=1
}

wait_for_api() {
  echo "Waiting for API on port $PORT ..."
  local deadline=$((SECONDS + READY_TIMEOUT))
  until curl -fsS "$API_URL/admin/router/health" >/dev/null 2>&1; do
    if [ $SECONDS -ge $deadline ]; then
      echo "API did not become ready; check $LOG_FILE"
      exit 1
    fi
    sleep 1
  done
  echo "API is ready."
}

post_json() {
  local path="$1"
  local body="$2"
  curl -fsS -H "Content-Type: application/json" -X POST "$API_URL$path" -d "$body"
}

register_adapters() {
  echo "Registering Ollama adapters..."
  post_json "/admin/router/adapters" "{\"name\":\"ollama-rights\",\"type\":\"ollama\",\"model\":\"$CRITIC_MODEL\"}" >/dev/null
  post_json "/admin/router/adapters" "{\"name\":\"ollama-fairness\",\"type\":\"ollama\",\"model\":\"$CRITIC_MODEL\"}" >/dev/null
  post_json "/admin/router/adapters" "{\"name\":\"ollama-risk\",\"type\":\"ollama\",\"model\":\"$CRITIC_MODEL\"}" >/dev/null
  post_json "/admin/router/adapters" "{\"name\":\"ollama-truth\",\"type\":\"ollama\",\"model\":\"$CRITIC_MODEL\"}" >/dev/null
  post_json "/admin/router/adapters" "{\"name\":\"ollama-pragmatics\",\"type\":\"ollama\",\"model\":\"$CRITIC_MODEL\"}" >/dev/null
  post_json "/admin/router/adapters" "{\"name\":\"ollama-phi3\",\"type\":\"ollama\",\"model\":\"$AUTONOMY_MODEL\"}" >/dev/null
}

bind_critics() {
  echo "Binding critics to adapters..."
  post_json "/admin/critics/bindings" "{\"critic\":\"rights\",\"adapter\":\"ollama-rights\"}" >/dev/null
  post_json "/admin/critics/bindings" "{\"critic\":\"fairness\",\"adapter\":\"ollama-fairness\"}" >/dev/null
  post_json "/admin/critics/bindings" "{\"critic\":\"risk\",\"adapter\":\"ollama-risk\"}" >/dev/null
  post_json "/admin/critics/bindings" "{\"critic\":\"truth\",\"adapter\":\"ollama-truth\"}" >/dev/null
  post_json "/admin/critics/bindings" "{\"critic\":\"pragmatics\",\"adapter\":\"ollama-pragmatics\"}" >/dev/null
  post_json "/admin/critics/bindings" "{\"critic\":\"autonomy\",\"adapter\":\"ollama-phi3\"}" >/dev/null
}

print_status() {
  echo ""
  echo "Router health:"
  curl -fsS "$API_URL/admin/router/health" || true
  echo ""
  echo ""
  echo "Critic bindings:"
  curl -fsS "$API_URL/admin/critics/bindings" || true
}

main() {
  ensure_uvicorn

  if api_running; then
    echo "API already running at $API_URL — skipping start."
  else
    start_api
  fi

  wait_for_api
  register_adapters
  bind_critics
  print_status

  if [ "$STARTED_API" -eq 1 ]; then
    echo ""
    echo "API was started by this script (PID: $API_PID). Use 'kill $API_PID' to stop it."
    echo "Logs: $LOG_FILE"
  fi
}

main "$@"
