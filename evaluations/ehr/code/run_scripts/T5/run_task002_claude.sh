#!/usr/bin/env bash
# Run task_002_diuretic_demo with Claude Sonnet 4.6 via OpenRouter
set -euo pipefail

MIMIC_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$MIMIC_ROOT/../.venv"

source "$VENV/bin/activate"
export PYTHONPATH="$MIMIC_ROOT:${PYTHONPATH:-}"

MODEL="anthropic/claude-sonnet-4-6"
TASK="tasks/task_002_diuretic_demo"
OUTPUT="runs/task_002_diuretic_demo/claude46"

echo "=== Running $MODEL on $TASK ==="
cd "$MIMIC_ROOT"
python -m agent.runner \
  --task "$TASK" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --max-steps 50 \
  --temperature 0.0

echo "=== Done: $MODEL ==="
