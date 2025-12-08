#!/bin/bash
# Submit remaining evaluation jobs for the llama2 bank-per-client experiment.
set -euo pipefail

ROOT="/home/hice1/sbhansali8/scratch/FederatedScope"
CKPT="$ROOT/ckpts/full/llama2_composite_meta3_unlearn_bank_perclient.ckpt"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint $CKPT not found. Please finish training before running evaluations."
  exit 1
fi

cd "$ROOT"

GLOBAL_JOB=$(sbatch scripts/sbatch_eval_llama_unlearn_bank_perclient_global.sbatch)
echo "Submitted global evaluation job: $GLOBAL_JOB"

CLIENT_JOB=$(sbatch scripts/sbatch_eval_llama_unlearn_bank_perclient_clients.sbatch)
echo "Submitted client evaluation job: $CLIENT_JOB"
