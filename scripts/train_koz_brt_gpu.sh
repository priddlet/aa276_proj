#!/usr/bin/env bash
# Train DeepReach-MPC KOZ BRT on an NVIDIA GPU node → simulation_output/deepreach_mpc_koz_v3
#
# Interactive (on a GPU worker, not a login node):
#   cd ~/aa276_proj
#   export CONDA_ENV=proj          # or: source .venv/bin/activate
#   ./scripts/train_koz_brt_gpu.sh
#
# Background + log:
#   ./scripts/train_koz_brt_gpu.sh --background
#   tail -f simulation_output/deepreach_mpc_koz_v3/train.log
#
# Optional env:
#   DEEPREACH_EPOCHS=100000  DEEPREACH_COUNTER_END=75000  DEEPREACH_DEVICE=cuda
#   DEEPREACH_CHECKPOINT_DIR=.../simulation_output/deepreach_mpc_koz_v3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BACKGROUND=0
EXTRA_TRAIN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --background|-b) BACKGROUND=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) EXTRA_TRAIN_ARGS+=("$arg") ;;
  esac
done

# --- Python env: conda name, or project .venv, or active env ---
if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"

# --- GPU / CUDA preflight ---
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Request a GPU node (srun/sbatch/interactive GPU)." >&2
  exit 1
fi
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || nvidia-smi
echo ""

"$PYTHON" - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    print("ERROR: PyTorch not installed:", e, file=sys.stderr)
    print("  pip install -r requirements-deepreach.txt", file=sys.stderr)
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
    print(f"  torch {torch.__version__}", file=sys.stderr)
    print("  Install CUDA build of PyTorch in THIS env (see requirements-deepreach.txt).", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: device 0 = {torch.cuda.get_device_name(0)}")
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
PY

# --- Training config (v3) ---
export DEEPREACH_CHECKPOINT_DIR="${DEEPREACH_CHECKPOINT_DIR:-$ROOT/simulation_output/deepreach_mpc_koz_v3}"
export BRT_HORIZON_S="${BRT_HORIZON_S:-1800}"
export DEEPREACH_EPOCHS="${DEEPREACH_EPOCHS:-100000}"
export DEEPREACH_COUNTER_END="${DEEPREACH_COUNTER_END:-75000}"
export DEEPREACH_DEVICE="${DEEPREACH_DEVICE:-cuda}"

mkdir -p "$DEEPREACH_CHECKPOINT_DIR"
LOG="$DEEPREACH_CHECKPOINT_DIR/train.log"

echo "=== DeepReach KOZ BRT training ==="
echo "  out:          $DEEPREACH_CHECKPOINT_DIR"
echo "  epochs:       $DEEPREACH_EPOCHS"
echo "  counter_end:  $DEEPREACH_COUNTER_END"
echo "  python:       $(command -v "$PYTHON")"
echo "  log:          $LOG"
echo ""

_run_train() {
  cd "$ROOT"
  "$PYTHON" -m simulation.brt.train --force "${EXTRA_TRAIN_ARGS[@]}"
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  echo "Starting in background (nohup)…"
  nohup "$PYTHON" -m simulation.brt.train --force "${EXTRA_TRAIN_ARGS[@]}" >>"$LOG" 2>&1 &
  echo $! >"$DEEPREACH_CHECKPOINT_DIR/train.pid"
  echo "PID $(cat "$DEEPREACH_CHECKPOINT_DIR/train.pid") — tail -f $LOG"
else
  _run_train 2>&1 | tee -a "$LOG"
fi
