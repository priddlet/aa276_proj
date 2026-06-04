#!/usr/bin/env bash
# Bootstrap a fresh GCP GPU VM for aa276_proj v3 training.
# Run once after: git clone ... && cd aa276_proj
#
#   chmod +x scripts/setup_gcp_gpu_vm.sh
#   ./scripts/setup_gcp_gpu_vm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-proj}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "=== aa276_proj GCP GPU setup ==="
echo "  repo:  $ROOT"
echo "  env:   $CONDA_ENV_NAME"
echo ""

# --- GPU driver (install manually first on plain Ubuntu) ---
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not working." >&2
  echo "  On a new Ubuntu GCP VM, install the driver first:" >&2
  echo "    sudo $ROOT/scripts/install_nvidia_driver_ubuntu.sh" >&2
  echo "    sudo reboot" >&2
  echo "  See docs/GCP_GPU_VM_SETUP.md" >&2
  exit 1
fi
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# --- conda ---
if ! command -v conda &>/dev/null; then
  echo "Installing Miniconda…"
  MINI="$HOME/miniconda3"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINI"
  # shellcheck source=/dev/null
  source "$MINI/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Conda env '$CONDA_ENV_NAME' exists — activating"
  conda activate "$CONDA_ENV_NAME"
else
  echo "Creating conda env '$CONDA_ENV_NAME' (python $PYTHON_VERSION)…"
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
  conda activate "$CONDA_ENV_NAME"
fi

echo "Python: $(which python) ($(python --version))"
echo ""

# --- dependencies ---
echo "=== pip install (CUDA torch + project deps) ==="
python -m pip install -U pip wheel
python -m pip install torch --index-url "$TORCH_INDEX"
python -m pip install -r "$ROOT/requirements-deepreach.txt"
python -m pip install -r "$ROOT/requirements.txt"

echo ""
echo "=== CUDA check ==="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda still False — fix GPU driver or torch install"
print("OK:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
PY

mkdir -p "$ROOT/simulation_output/deepreach_mpc_koz_v3"
chmod +x "$ROOT/scripts/train_koz_brt_gpu.sh" "$ROOT/scripts/retrain_koz_brt_v3.sh" 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Start training:"
echo "  cd $ROOT"
echo "  export CONDA_ENV=$CONDA_ENV_NAME"
echo "  ./scripts/train_koz_brt_gpu.sh"
echo ""
echo "Or background:"
echo "  ./scripts/train_koz_brt_gpu.sh --background"
echo "  tail -f simulation_output/deepreach_mpc_koz_v3/train.log"
