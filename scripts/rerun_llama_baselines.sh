#!/usr/bin/env bash
# Relaunch the LLaMA/Qwen baseline training array and the dependent eval jobs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf "[rerun-baselines] %s\n" "$*" >&2
}

log "Submitting baseline training array (llama + qwen)..."
BASELINES_JOB=$(sbatch --parsable sbatch_train_baselines_full.sbatch)
log "Baseline array submitted: ${BASELINES_JOB}"

log "Queuing llama global eval (arrays 0-2) after baseline completion..."
sbatch --parsable --dependency=afterok:${BASELINES_JOB} --array=0-2 sbatch_eval_global_full.sbatch

log "Queuing llama client eval (arrays 0-8) after baseline completion..."
sbatch --parsable --dependency=afterok:${BASELINES_JOB} --array=0-8 sbatch_eval_clients_full.sbatch

log "Queuing baseline eval array after baseline completion..."
sbatch --parsable --dependency=afterok:${BASELINES_JOB} sbatch_eval_baselines_full.sbatch

log "Done. Monitor with 'squeue -u $USER'."
