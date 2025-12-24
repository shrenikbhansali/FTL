#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_tulu_pipeline.sh [options]

Options:
  --dry-run   Print the sbatch commands without submitting.
  --skip-eval Do not launch evaluation array.
  -h, --help  Show this message.
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
  printf "[tulu-pipeline] %s\n" "$*" >&2
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

run mkdir -p logs results_tulu

log "Submitting Tulu training jobs..."
FEDAVG_JOB=$(submit_sbatch "train-tulu-fedavg" "$SBATCH_DIR/sbatch_train_tulu_fedavg.sbatch")
BANK_JOB=$(submit_sbatch "train-tulu-bank-perclient" "$SBATCH_DIR/sbatch_train_tulu_bank_perclient.sbatch")
CENTRAL_JOB=$(submit_sbatch "train-tulu-centralized" "$SBATCH_DIR/sbatch_train_tulu_centralized.sbatch")

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Submitting Tulu evaluations..."
  submit_sbatch "eval-tulu-fedavg" --dependency=afterok:${FEDAVG_JOB} --array=0-3 \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch"
  submit_sbatch "eval-tulu-bank" --dependency=afterok:${BANK_JOB} --array=4-7 \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch"
  submit_sbatch "eval-tulu-central" --dependency=afterok:${CENTRAL_JOB} --array=8-11 \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch"
  submit_sbatch "eval-tulu-llama2-base" --array=12-15 \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch"
else
  log "Skipping evaluations."
fi

log "Tulu pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
