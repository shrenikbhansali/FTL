#!/bin/bash
#SBATCH --job-name=wandb-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=tulupipe_logs/wandb_smoke_%A.out
#SBATCH --error=tulupipe_logs/wandb_smoke_%A.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm

cd /home/hice1/sbhansali8/scratch/FederatedScope
mkdir -p tulupipe_logs

export TMPDIR="$PWD/tulupipe_smoke/tmp"
export WANDB_DIR="$PWD/tulupipe_smoke/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"

# Disable the service backend to avoid socket issues on /tmp.
export WANDB_DISABLE_SERVICE=1

python scripts/wandb_smoke_test.py
