#!/bin/bash
#SBATCH --job-name=txagent_task002
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=txagent_slurm_%j.out
#SBATCH --error=txagent_slurm_%j.err
# NOTE: --output/--error are relative to the submission directory (sbatch
# does not expand shell variables in #SBATCH directives). Submit this script
# from $MIMIC_ROOT/runs/task_002_diuretic_demo/, or edit the paths above.
# Set --account to your own cluster allocation.
#SBATCH --account=xu_hua

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Set MIMIC_ROOT / VENV to your local paths, or export them before sbatch
# submission (sbatch does not inherit interactive shell env by default
# unless --export=ALL or --get-user-env is used).
MIMIC_ROOT="${MIMIC_ROOT:-$HOME/mimic-main}"
VENV="${VENV:-$HOME/.venv}"
PYTHON="$VENV/bin/python3"

# ── CUDA ────────────────────────────────────────────────────────────────────
module load CUDA/12.2.2 2>/dev/null || \
module load CUDA/12.2.0 2>/dev/null || \
module load CUDA 2>/dev/null || true

if [[ -z "${CUDA_HOME:-}" ]]; then
    for p in /gpfs/radev/apps/avx512/software/CUDA/12.2.2 \
              /usr/local/cuda-12.2 /usr/local/cuda; do
        [[ -d "$p" ]] && export CUDA_HOME="$p" && break
    done
fi
echo "CUDA_HOME: ${CUDA_HOME:-not set}"

CUDA_DRIVER_DIR=$(find /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib64 \
    -name 'libcuda.so.1' 2>/dev/null | head -1 | xargs -I{} dirname {} 2>/dev/null)
if [[ -n "$CUDA_DRIVER_DIR" ]]; then
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_DRIVER_DIR}:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
fi

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi N/A"

# ── Check CUDA/torch ─────────────────────────────────────────────────────────
$PYTHON -c "
import torch, sys
print(f'torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available(): sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ── TxAgent paths ────────────────────────────────────────────────────────────
# Set TXAGENT_SRC to your local TxAgent checkout's src/ directory.
TXAGENT_SRC="${TXAGENT_SRC:-$MIMIC_ROOT/../EHR_bench/agent_deployments/TxAgent/src}"
export PYTHONPATH="$MIMIC_ROOT:$TXAGENT_SRC:${PYTHONPATH:-}"

# HuggingFace cache (model already downloaded)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# ── Fix numpy: tensorflow 2.13 was compiled with numpy 1.x; downgrade if needed ──
NUMPY_OK=$($PYTHON -c "import numpy as np; v=tuple(int(x) for x in np.__version__.split('.')[:2]); print('yes' if v < (2,0) else 'no')" 2>/dev/null)
if [[ "$NUMPY_OK" != "yes" ]]; then
    echo "[INFO] Downgrading numpy to <2.0 (required for tensorflow/vllm compat) ..."
    $PYTHON -m pip install "numpy<2.0" -q
fi

# ── Run TxAgent on task_002 ──────────────────────────────────────────────────
echo ""
echo "Running TxAgent on task_002_diuretic_demo ..."
cd "$MIMIC_ROOT"

$PYTHON -m agent.txagent_runner \
    --task tasks/task_002_diuretic_demo \
    --output runs/task_002_diuretic_demo/txagent \
    --model-name "mims-harvard/TxAgent-T1-Llama-3.1-8B" \
    --rag-model-name "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B" \
    --max-round 20 \
    --max-new-tokens 1024 \
    --temperature 0.3 \
    --vllm-enforce-eager \
    --vllm-gpu-memory-utilization 0.85 \
    --vllm-max-model-len 32768 \
    --disable-tool-rag \
    2>&1 | tee runs/task_002_diuretic_demo/txagent_run.log

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
