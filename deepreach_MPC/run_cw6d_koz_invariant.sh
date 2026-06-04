#!/usr/bin/env bash
# Retrain 6D CW KOZ avoid-BRT with explicit KOZ interior constraint at random t.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DEEPREACH_CHECKPOINT_DIR="${DEEPREACH_CHECKPOINT_DIR:-$ROOT/simulation_output/deepreach_mpc_koz_v3}"
export BRT_HORIZON_S="${BRT_HORIZON_S:-1800}"
export DEEPREACH_COUNTER_END="${DEEPREACH_COUNTER_END:-75000}"
export DEEPREACH_KOZ_SAMPLES="${DEEPREACH_KOZ_SAMPLES:-2500}"
export DEEPREACH_KOZ_LOSS_WEIGHT="${DEEPREACH_KOZ_LOSS_WEIGHT:-25}"
python -m simulation.brt.train --force "$@"
