#!/usr/bin/env bash
set -euo pipefail

# Demo-friendly starter for local/Ollama models.
# Binds critics to your local images and launches the API on a demo port.

PORT="${PORT:-8001}"
CONFIG="${ELEANOR_CONFIG:-./config/eleanor.yaml}"

# Core env
export ELEANOR_ENV="${ELEANOR_ENV:-development}"
export ELEANOR_CONFIG="$CONFIG"

# Ollama models to register (adapters become ollama-<name>)
export OLLAMA_MODELS="${OLLAMA_MODELS:-eleanor-pragmatics,eleanor-truth,eleanor-risk,eleanor-fairness,eleanor-rights,mistral,phi3,llama3}"

# Critic → adapter bindings (customize as needed)
export CRITIC_MODEL_BINDINGS="${CRITIC_MODEL_BINDINGS:-{
  \"pragmatics\": \"ollama-eleanor-pragmatics\",
  \"truth\": \"ollama-eleanor-truth\",
  \"risk\": \"ollama-eleanor-risk\",
  \"fairness\": \"ollama-eleanor-fairness\",
  \"rights\": \"ollama-eleanor-rights\",
  \"autonomy\": \"ollama-phi3\",
  \"privacy_identity\": \"ollama-mistral\"
}}"

# Fallback default adapter for any unlisted critic
export CRITIC_DEFAULT_ADAPTER="${CRITIC_DEFAULT_ADAPTER:-ollama-phi3}"
export ELEANOR_DEFAULT_MODEL="${ELEANOR_DEFAULT_MODEL:-$CRITIC_DEFAULT_ADAPTER}"

echo "🧠 Starting ELEANOR V8 demo"
echo "Config: $ELEANOR_CONFIG"
echo "Port:   $PORT"
echo "Models: $OLLAMA_MODELS"
echo "Bindings: $CRITIC_MODEL_BINDINGS"

command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
python3 -m uvicorn api.rest.main:app --host 0.0.0.0 --port "$PORT" --lifespan on --log-level info
