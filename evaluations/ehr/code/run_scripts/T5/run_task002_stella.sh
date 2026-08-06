#!/usr/bin/env bash
# Run task_002_diuretic_demo with STELLA (multi-agent, OpenRouter)
set -euo pipefail

MIMIC_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$MIMIC_ROOT/../.venv}"
# Set STELLA_DIR to your local STELLA checkout; defaults to a sibling
# EHR_bench/agent_deployments/STELLA directory relative to this repo.
STELLA_DIR="${STELLA_DIR:-$MIMIC_ROOT/../EHR_bench/agent_deployments/STELLA}"

source "$VENV/bin/activate"
export PYTHONPATH="$MIMIC_ROOT:${PYTHONPATH:-}"
export STELLA_DIR="$STELLA_DIR"

TASK="tasks/task_002_diuretic_demo"
OUTPUT="runs/task_002_diuretic_demo/stella"

echo "=== Running STELLA on $TASK ==="
cd "$MIMIC_ROOT"
python -m agent.stella_runner \
  --task "$TASK" \
  --output "$OUTPUT" \
  --stella-dir "$STELLA_DIR" \
  --no-template \
  --disable-mem0 \
  --no-web-tools \
  --timeout 3600 \
  2>&1 | tee "$OUTPUT/../stella_run.log" || true

echo "=== Done: STELLA ==="
