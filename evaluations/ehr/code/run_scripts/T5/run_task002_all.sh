#!/usr/bin/env bash
# Master script: run all 4 models on task_002_diuretic_demo sequentially
# Models: GPT-5.2, Gemini-2.5-Pro, Claude-Sonnet-4.6, STELLA
set -euo pipefail

MIMIC_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "====================================================="
echo " task_002_diuretic_demo — full 4-model run"
echo " Start: $(date)"
echo "====================================================="

# Create log directory
mkdir -p "$MIMIC_ROOT/runs/task_002_diuretic_demo"

run_model() {
  local script="$1"
  local name="$2"
  echo ""
  echo ">>> [$name] Starting at $(date)"
  bash "$MIMIC_ROOT/$script" && echo ">>> [$name] Done at $(date)" \
    || echo ">>> [$name] FAILED at $(date) — check log"
}

run_model "run_task002_gpt52.sh"   "GPT-5.2"
run_model "run_task002_gemini.sh"  "Gemini-2.5-Pro"
run_model "run_task002_claude.sh"  "Claude-Sonnet-4.6"
run_model "run_task002_stella.sh"  "STELLA"

echo ""
echo "====================================================="
echo " All models complete: $(date)"
echo " Run outputs: $MIMIC_ROOT/runs/task_002_diuretic_demo/"
echo "====================================================="
