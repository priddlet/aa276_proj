#!/usr/bin/env bash
# Train 6D CW KOZ avoid-BRT via simulation wrapper (100k epochs, exact model, MPC on).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DEEPREACH_CHECKPOINT_DIR="${DEEPREACH_CHECKPOINT_DIR:-$ROOT/simulation_output/deepreach_mpc_koz_v2}"
export BRT_HORIZON_S="${BRT_HORIZON_S:-1800}"
python -m simulation.brt.train --force "$@"
