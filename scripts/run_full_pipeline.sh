#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_GLOBAL_EVAL=0
SKIP_CLIENT_EVAL=0
SKIP_BASELINE_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-global-eval)
      SKIP_GLOBAL_EVAL=1
      shift
      ;;
    --skip-client-eval)
      SKIP_CLIENT_EVAL=1
      shift
      ;;
    --skip-baseline-eval)
      SKIP_BASELINE_EVAL=1
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_full_pipeline.sh [options]

Options:
  --dry-run             Print the sbatch commands without submitting.
  --skip-global-eval    Do not launch global evaluation array.
  --skip-client-eval    Do not launch per-client evaluation array.
  --skip-baseline-eval  Do not launch baseline evaluation array.
  -h, --help            Show this message.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Run this script on a Slurm login node." >&2
  exit 1
fi

log() {
  printf "[full-pipeline] %s\n" "$*" >&2
}

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN: $*"
  else
    "$@"
  fi
}

submit_sbatch() {
  local label=$1
  shift
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN sbatch ($label): $*"
    echo "dry-$label"
    return
  fi
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "$label job submitted: $job_id"
  echo "$job_id"
}

run mkdir -p logs results_full

log "Submitting full-scale training jobs..."
LLAMA_JOB=$(submit_sbatch "train-llama-full" sbatch_train_llama_full.sbatch)
QWEN_JOB=$(submit_sbatch "train-qwen-full" sbatch_train_qwen_moe_full.sbatch)
BASE_JOB=$(submit_sbatch "train-baselines-full" sbatch_train_baselines_full.sbatch)

if [[ $SKIP_GLOBAL_EVAL -eq 0 ]]; then
  log "Submitting global evaluations..."
  submit_sbatch "eval-global-full-llama" --dependency=afterok:${LLAMA_JOB} --array=0-2 sbatch_eval_global_full.sbatch
  submit_sbatch "eval-global-full-qwen" --dependency=afterok:${QWEN_JOB} --array=3-5 sbatch_eval_global_full.sbatch
else
  log "Skipping global evaluations."
fi

if [[ $SKIP_CLIENT_EVAL -eq 0 ]]; then
  log "Submitting client evaluations..."
  submit_sbatch "eval-clients-full-llama" --dependency=afterok:${LLAMA_JOB} --array=0-8 sbatch_eval_clients_full.sbatch
  submit_sbatch "eval-clients-full-qwen" --dependency=afterok:${QWEN_JOB} --array=9-17 sbatch_eval_clients_full.sbatch
else
  log "Skipping client evaluations."
fi

if [[ $SKIP_BASELINE_EVAL -eq 0 ]]; then
  log "Submitting baseline evaluations..."
  submit_sbatch "eval-baselines-full" --dependency=afterok:${BASE_JOB} sbatch_eval_baselines_full.sbatch
else
  log "Skipping baseline evaluations."
fi

log "Full pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
