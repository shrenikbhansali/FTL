#!/bin/bash
#SBATCH --job-name=fsllm-collect-tulu-metrics
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
export WANDB_API_KEY=78575c71e3200c01b61cae0272aef2d952ca18f9
if [[ -z "${WANDB_ENTITY:-}" ]]; then
  export WANDB_ENTITY=sbhansali8-georgia-institute-of-technology
fi
if [[ -z "${WANDB_PROJECT:-}" ]]; then
  export WANDB_PROJECT=FTL-Dev
fi
export TMPDIR="${TULU_RESULTS_DIR:-$PWD/results_tulu}/tmp"
export WANDB_DIR="${TULU_RESULTS_DIR:-$PWD/results_tulu}/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
unset WANDB_SERVICE
cd /home/hice1/sbhansali8/scratch/FederatedScope

export WANDB_DISABLE_SERVICE=1
if [[ "${TULU_EXP:-}" != "bank_perclient" ]]; then
  unset WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG
fi

python scripts/collect_tulu_eval_results.py \
  --results-root "${TULU_RESULTS_DIR:-$PWD/results_tulu}" \
  --exp "${TULU_EXP:-}" \
  --eval-job-id "${TULU_EVAL_JOB_ID:-}" \
  --fedavg-eval-job-id "${TULU_FEDAVG_EVAL_JOB_ID:-}"
