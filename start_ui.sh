#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm (Node 18+) is required"; exit 1; }

# Local defaults
export ELEANOR_ENVIRONMENT="${ELEANOR_ENVIRONMENT:-${ELEANOR_ENV:-development}}"
export ELEANOR_ENV="${ELEANOR_ENV:-$ELEANOR_ENVIRONMENT}"
export AUTH_ENABLED=${AUTH_ENABLED:-false}
export JWT_SECRET=${JWT_SECRET:-dev-secret}
export ELEANOR_ENABLE_ADMIN_WRITE=${ELEANOR_ENABLE_ADMIN_WRITE:-true}
export ELEANOR_CONFIG=${ELEANOR_CONFIG:-./config/eleanor.yaml}
export OPA_URL=${OPA_URL:-http://localhost:8181}
export OPA_POLICY_PATH=${OPA_POLICY_PATH:-v1/data/eleanor/decision}
export OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.1:8b}
if [ -z "${OLLAMA_MODELS:-}" ] && command -v ollama >/dev/null 2>&1; then
  detected_models=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')
  if [ -n "$detected_models" ]; then
    OLLAMA_MODELS=$(printf "%s\n" "$detected_models" | paste -sd "," -)
  fi
fi
export OLLAMA_MODELS=${OLLAMA_MODELS:-"llama3.1:8b,eleanor-rights:latest,eleanor-fairness:latest,eleanor-truth:latest,eleanor-risk:latest,eleanor-pragmatics:latest"}
export CRITIC_MODEL_BINDINGS=${CRITIC_MODEL_BINDINGS:-'{"rights":"ollama-eleanor-rights:latest","fairness":"ollama-eleanor-fairness:latest","truth":"ollama-eleanor-truth:latest","risk":"ollama-eleanor-risk:latest","operations":"ollama-eleanor-pragmatics:latest","autonomy":"ollama-llama3.1:8b"}'}

wait_for_url() {
  local url=$1
  local attempts=${2:-30}
  local delay=${3:-1}
  for _ in $(seq 1 "$attempts"); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "$url" >/dev/null 2>&1; then
        return 0
      fi
    else
      if python3 - <<PY >/dev/null 2>&1
import urllib.request, sys
try:
    urllib.request.urlopen("$url", timeout=1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
      then
        return 0
      fi
    fi
    sleep "$delay"
  done
  return 1
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

start_opa() {
  if [ "${ELEANOR_SKIP_OPA:-false}" = "true" ]; then
    echo "Skipping OPA startup (ELEANOR_SKIP_OPA=true)."
    return 0
  fi
  if wait_for_url "${OPA_URL}/health" 1 0; then
    return 0
  fi
  if command -v docker >/dev/null 2>&1; then
    echo "Starting OPA container..."
    docker rm -f opa >/dev/null 2>&1 || true
    docker run -d --name opa \
      -p 8181:8181 \
      -v "$ROOT/opa/policies":/policies:ro \
      openpolicyagent/opa:latest \
      run --server --addr=0.0.0.0:8181 /policies >/dev/null
  else
    echo "OPA not running and docker is unavailable."
    return 1
  fi
  if ! wait_for_url "${OPA_URL}/health" 30 1; then
    echo "❌ OPA did not become ready at ${OPA_URL}/health"
    return 1
  fi
  return 0
}

start_grafana() {
  if ! is_truthy "${ELEANOR_START_GRAFANA:-true}"; then
    return 0
  fi
  if command -v docker >/dev/null 2>&1; then
    local net_name="eleanor_obsv"
    local prom_config
    prom_config=$(mktemp -t eleanor-prom-XXXX.yml)
    cat > "$prom_config" <<'EOF'
global:
  scrape_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: "eleanor-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]

  - job_name: "opa"
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8181"]
EOF
    docker network inspect "$net_name" >/dev/null 2>&1 || docker network create "$net_name" >/dev/null
    docker rm -f prometheus grafana >/dev/null 2>&1 || true
    docker run -d --name prometheus --network "$net_name" -p 9090:9090 \
      -v "$prom_config":/etc/prometheus/prometheus.yml:ro \
      prom/prometheus:latest \
      --config.file=/etc/prometheus/prometheus.yml >/dev/null
    docker run -d --name grafana --network "$net_name" -p 3000:3000 \
      -e GF_SECURITY_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}" \
      -e GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}" \
      -e GF_SECURITY_ALLOW_EMBEDDING="true" \
      -e GF_AUTH_ANONYMOUS_ENABLED="true" \
      -e GF_AUTH_ANONYMOUS_ORG_ROLE="Viewer" \
      -v "$ROOT/grafana/provisioning":/etc/grafana/provisioning \
      grafana/grafana:latest >/dev/null
    if ! wait_for_url "http://localhost:3000" 30 1; then
      echo "❌ Grafana did not become ready at http://localhost:3000"
      return 1
    fi
    return 0
  fi
  echo "Grafana not started because docker is unavailable."
  return 1
}

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

start_opa
start_grafana

echo "Starting API + UI at http://localhost:8000/ui/ ..."
if command -v open >/dev/null 2>&1; then
  (sleep 2 && open "http://localhost:8000/ui/") >/dev/null 2>&1 &
fi
uvicorn api.rest.main:app --host 0.0.0.0 --port 8000
