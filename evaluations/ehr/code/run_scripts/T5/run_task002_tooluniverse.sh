#!/usr/bin/env bash
# Run task_002_diuretic_demo with GPT-5.2 + ToolUniverse bridge via OpenRouter
set -euo pipefail

MIMIC_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$MIMIC_ROOT/../.venv}"

source "$VENV/bin/activate"
export PYTHONPATH="$MIMIC_ROOT:${PYTHONPATH:-}"

# Add ToolUniverse src to PYTHONPATH. Set TOOLUNIVERSE_SRC to your local
# ToolUniverse checkout's src/ directory; defaults to a sibling
# EHR_bench/agent_deployments/ToolUniverse/src directory.
TU_SRC="${TOOLUNIVERSE_SRC:-$MIMIC_ROOT/../EHR_bench/agent_deployments/ToolUniverse/src}"
export PYTHONPATH="$TU_SRC:${PYTHONPATH}"
export TOOLUNIVERSE_LAZY_LOADING=true
export TOOLUNIVERSE_QUIET=1

MODEL="openai/gpt-5.2"
TASK="tasks/task_002_diuretic_demo"
OUTPUT="runs/task_002_diuretic_demo/tooluniverse"

echo "=== Running $MODEL + ToolUniverse on $TASK ==="
cd "$MIMIC_ROOT"
python -m agent.runner \
  --task "$TASK" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --max-steps 50 \
  --temperature 0.0 \
  --tooluniverse \
  2>&1 | tee "$OUTPUT/../tooluniverse_run.log" || true

echo "=== Done: ToolUniverse ==="
