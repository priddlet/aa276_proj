#!/usr/bin/env bash
# Retrain 6D CW KOZ avoid-BRT into simulation_output/deepreach_mpc_koz_v3.
# Requires: CUDA GPU, .venv with torch + deepreach deps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi
exec "$ROOT/deepreach_MPC/run_cw6d_koz.sh" "$@"
