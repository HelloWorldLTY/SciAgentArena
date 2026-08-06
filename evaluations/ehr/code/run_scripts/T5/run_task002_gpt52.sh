#!/usr/bin/env bash
# Run task_002_diuretic_demo with GPT-5.2 via OpenRouter
set -euo pipefail

MIMIC_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$MIMIC_ROOT/../.venv"

source "$VENV/bin/activate"
export PYTHONPATH="$MIMIC_ROOT:${PYTHONPATH:-}"

MODEL="openai/gpt-5.2"
TASK="tasks/task_002_diuretic_demo"
OUTPUT="runs/task_002_diuretic_demo/gpt52"

echo "=== Running $MODEL on $TASK ==="
cd "$MIMIC_ROOT"
python -m agent.runner \
  --task "$TASK" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --max-steps 50 \
  --temperature 0.0 \
  2>&1 | tee "$OUTPUT/../gpt52_run.log" || true

echo "=== Done: $MODEL ==="
