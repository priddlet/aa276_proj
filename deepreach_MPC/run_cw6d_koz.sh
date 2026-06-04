#!/usr/bin/env bash
# Train 6D CW KOZ avoid-BRT via simulation wrapper (100k epochs, exact model, MPC on).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DEEPREACH_CHECKPOINT_DIR="${DEEPREACH_CHECKPOINT_DIR:-$ROOT/simulation_output/deepreach_mpc_koz_v3}"
export BRT_HORIZON_S="${BRT_HORIZON_S:-1800}"
EPOCHS="${DEEPREACH_EPOCHS:-100000}"
export DEEPREACH_EPOCHS="$EPOCHS"
# Full 1800 s horizon by epoch 75k; LR decay after counter_end + pretrain (~epoch 76k).
export DEEPREACH_COUNTER_END="${DEEPREACH_COUNTER_END:-75000}"
echo "Training KOZ BRT → ${DEEPREACH_CHECKPOINT_DIR} (epochs=${EPOCHS}, counter_end=${DEEPREACH_COUNTER_END})"
python -m simulation.brt.train --force "$@"
