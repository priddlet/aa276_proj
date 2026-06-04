#!/usr/bin/env bash
# Retrain v3 — wrapper around train_koz_brt_gpu.sh (CUDA required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/train_koz_brt_gpu.sh" "$@"
