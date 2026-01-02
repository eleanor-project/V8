#!/usr/bin/env bash
set -e

echo "🧠 Starting ELEANOR — Ethical Living Engine for Normative Oversight & Reasoning"
echo "------------------------------------------------------------"

# ---- CONFIG ----
export ELEANOR_ENV="${ELEANOR_ENV:-local}"
export ELEANOR_ENVIRONMENT="${ELEANOR_ENVIRONMENT:-$ELEANOR_ENV}"
export ELEANOR_LOG_LEVEL=INFO
export ELEANOR_CONFIG=./config/eleanor.yaml

export PYTHONUNBUFFERED=1

# Optional model defaults (router can override)
export ELEANOR_DEFAULT_MODEL=auto
export ELEANOR_ENABLE_STREAMING=true

# Vector store flags
export ELEANOR_PRECEDENT_BACKEND=weaviate
export ELEANOR_ENABLE_PRECEDENT_WRITEBACK=true

# Governance
export ELEANOR_GOVERNANCE_MODE=constitutional
export ELEANOR_EVIDENCE_RECORDER=true

# ---- SANITY CHECKS ----
command -v python3 >/dev/null 2>&1 || {
  echo "❌ python3 not found"
  exit 1
}

if [ ! -f "$ELEANOR_CONFIG" ]; then
  echo "❌ Missing config file: $ELEANOR_CONFIG"
  exit 1
fi

echo "✅ Environment OK"
echo "📄 Config: $ELEANOR_CONFIG"
echo "🤖 Default Model: $ELEANOR_DEFAULT_MODEL"

# ---- START API ----
echo ""
echo "🚀 Launching Eleanor API..."

python3 -m orchestrator.api \
  --config "$ELEANOR_CONFIG"
