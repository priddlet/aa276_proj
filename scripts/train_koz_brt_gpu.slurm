#!/bin/bash
#SBATCH --job-name=koz-brt-v3
#SBATCH --output=simulation_output/deepreach_mpc_koz_v3/slurm_%j.log
#SBATCH --error=simulation_output/deepreach_mpc_koz_v3/slurm_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Cluster example — edit partition/account for your site, then:
#   mkdir -p simulation_output/deepreach_mpc_koz_v3
#   sbatch scripts/train_koz_brt_gpu.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/aa276_proj}"

# Activate your GPU conda env (change name if needed)
export CONDA_ENV="${CONDA_ENV:-proj}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export DEEPREACH_CHECKPOINT_DIR="${SLURM_SUBMIT_DIR}/simulation_output/deepreach_mpc_koz_v3"
export DEEPREACH_EPOCHS="${DEEPREACH_EPOCHS:-100000}"
export DEEPREACH_COUNTER_END="${DEEPREACH_COUNTER_END:-75000}"

bash scripts/train_koz_brt_gpu.sh
