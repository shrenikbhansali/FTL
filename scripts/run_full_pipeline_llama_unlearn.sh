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
Usage: run_full_pipeline_llama_unlearn.sh [options]

Options:
  --dry-run             Print the sbatch commands without submitting.
  --skip-global-eval    Do not launch global evaluation arrays.
  --skip-client-eval    Do not launch per-client evaluation arrays.
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
SBATCH_DIR="$ROOT_DIR/scripts"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Run this script on a Slurm login node." >&2
  exit 1
fi

log() {
  printf "[llama-unlearn-pipeline] %s\n" "$*" >&2
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

UNLEARN_SBATCH="$SBATCH_DIR/sbatch_train_llama_unlearn_full.sbatch"
if [[ ! -f "$UNLEARN_SBATCH" ]]; then
  log "ERROR: Unlearn training sbatch not found at $UNLEARN_SBATCH"
  exit 1
fi

log "Submitting full-scale llama training jobs..."
LLAMA_JOB=""
LLAMA_SBATCH="$SBATCH_DIR/sbatch_train_llama_full.sbatch"
if [[ -f "$LLAMA_SBATCH" ]]; then
  LLAMA_JOB=$(submit_sbatch "train-llama-full" "$LLAMA_SBATCH")
else
  log "Standard LLaMA training sbatch missing ($LLAMA_SBATCH); skipping standard training/evals."
fi

UNLEARN_JOB=$(submit_sbatch "train-llama-unlearn-full" "$UNLEARN_SBATCH")

BASE_JOB=""
BASE_SBATCH="$SBATCH_DIR/sbatch_train_baselines_full.sbatch"
if [[ -f "$BASE_SBATCH" ]]; then
  BASE_JOB=$(submit_sbatch "train-baselines-full" "$BASE_SBATCH")
else
  log "Baseline training sbatch missing ($BASE_SBATCH); skipping baseline training/evals."
fi

if [[ $SKIP_GLOBAL_EVAL -eq 0 ]]; then
  log "Submitting global evaluations..."
  if [[ -n "$LLAMA_JOB" ]]; then
    submit_sbatch "eval-global-full-llama" --dependency=afterok:${LLAMA_JOB} --array=0-2 "$SBATCH_DIR/sbatch_eval_global_full.sbatch"
  else
    log "Skipping standard global evaluation; no corresponding training job."
  fi
  submit_sbatch "eval-global-unlearn" --dependency=afterok:${UNLEARN_JOB} "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"
else
  log "Skipping global evaluations."
fi

if [[ $SKIP_CLIENT_EVAL -eq 0 ]]; then
  log "Submitting client evaluations..."
  if [[ -n "$LLAMA_JOB" ]]; then
    submit_sbatch "eval-clients-full-llama" --dependency=afterok:${LLAMA_JOB} --array=0-8 "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"
  else
    log "Skipping standard client evaluation; no corresponding training job."
  fi
  submit_sbatch "eval-clients-unlearn" --dependency=afterok:${UNLEARN_JOB} "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"
else
  log "Skipping client evaluations."
fi

if [[ $SKIP_BASELINE_EVAL -eq 0 ]]; then
  log "Submitting baseline evaluations..."
  if [[ -n "$BASE_JOB" ]]; then
    submit_sbatch "eval-baselines-full" --dependency=afterok:${BASE_JOB} "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"
  else
    log "Skipping baseline evaluations; no baseline training job."
  fi
else
  log "Skipping baseline evaluations."
fi

log "LLaMA + UNLEARN pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
