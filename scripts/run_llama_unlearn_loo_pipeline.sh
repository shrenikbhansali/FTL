#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_GLOBAL_EVAL=0
SKIP_CLIENT_EVAL=0

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
    -h|--help)
      cat <<'USAGE'
Usage: run_llama_unlearn_loo_pipeline.sh [options]

Launch the LLaMA-2 UNLEARN (LOO) training arrays plus the dependent
global/client evaluation arrays.

Options:
  --dry-run           Print the sbatch commands without submitting jobs.
  --skip-global-eval  Do not launch the global evaluation jobs.
  --skip-client-eval  Do not launch the client evaluation jobs.
  -h, --help          Show this message and exit.
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
  echo "[loo-pipeline] ERROR: sbatch not found in PATH; run on a Slurm login node." >&2
  exit 1
fi

log() {
  printf "[loo-pipeline] %s\n" "$*" >&2
}

run_cmd() {
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
    echo "dry-${label}"
    return
  fi
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "$label job submitted: $job_id"
  echo "$job_id"
}

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    log "ERROR: missing $label at $path"
    exit 1
  fi
}

run_cmd mkdir -p logs results_full

TRAIN_SBATCH="$ROOT_DIR/scripts/sbatch_train_llama_unlearn_loo_full.sbatch"
EVAL_GLOBAL_SBATCH="$ROOT_DIR/scripts/sbatch_eval_unlearn_global_loo_full.sbatch"
EVAL_CLIENT_SBATCH="$ROOT_DIR/scripts/sbatch_eval_unlearn_clients_loo_full.sbatch"

require_file "$TRAIN_SBATCH" "LLaMA LOO training sbatch"
require_file "$EVAL_GLOBAL_SBATCH" "LLaMA LOO global evaluation sbatch"
require_file "$EVAL_CLIENT_SBATCH" "LLaMA LOO client evaluation sbatch"

log "Submitting LLaMA-2 UNLEARN (LOO) training arrays..."
TRAIN_JOB=$(submit_sbatch "train-llama-unlearn-loo" "$TRAIN_SBATCH")

if [[ $SKIP_GLOBAL_EVAL -eq 0 ]]; then
  log "Submitting dependent global evaluations..."
  submit_sbatch "eval-loo-global" --dependency=afterok:${TRAIN_JOB} "$EVAL_GLOBAL_SBATCH"
else
  log "Skipping global evaluations."
fi

if [[ $SKIP_CLIENT_EVAL -eq 0 ]]; then
  log "Submitting dependent client evaluations..."
  submit_sbatch "eval-loo-clients" --dependency=afterok:${TRAIN_JOB} "$EVAL_CLIENT_SBATCH"
else
  log "Skipping client evaluations."
fi

log "LOO pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
