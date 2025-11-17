#!/usr/bin/env bash
set -euo pipefail

# run.sh — activate venv, export LLM vars, and start the app
# Usage:
#   Make executable once: chmod +x run.sh
#   Override defaults by exporting env vars or passing MODEL as first arg.

: "${LLM_URL:=http://127.0.0.1:1234/v1/chat/completions}"
: "${MODEL_NAME:=dolphin-2.9.3-mistral-nemo-12b}"
VENV="${VENV:-.venv}"

# Allow passing model name as first argument
if [ "$#" -ge 1 ]; then
  MODEL_NAME="$1"
fi

if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

export LLM_URL MODEL_NAME

echo "Starting app with LLM_URL=$LLM_URL MODEL_NAME=$MODEL_NAME"
python app.py
