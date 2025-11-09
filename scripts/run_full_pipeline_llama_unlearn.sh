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

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    log "ERROR: Missing $label at $path"
    exit 1
  fi
}

LLAMA_SBATCH="$SBATCH_DIR/sbatch_train_llama_full.sbatch"
LLAMA_UNLEARN_SBATCH="$SBATCH_DIR/sbatch_train_llama_unlearn_full.sbatch"
QWEN_SBATCH="$SBATCH_DIR/sbatch_train_qwen_moe_full.sbatch"
QWEN_UNLEARN_SBATCH="$SBATCH_DIR/sbatch_train_qwen_unlearn_full.sbatch"
BASE_SBATCH="$SBATCH_DIR/sbatch_train_baselines_full.sbatch"
EVAL_GLOBAL_SBATCH="$SBATCH_DIR/sbatch_eval_global_full.sbatch"
EVAL_CLIENT_SBATCH="$SBATCH_DIR/sbatch_eval_clients_full.sbatch"
EVAL_BASE_SBATCH="$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"
EVAL_UNLEARN_GLOBAL_SBATCH="$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"
EVAL_UNLEARN_CLIENT_SBATCH="$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

require_file "$LLAMA_SBATCH" "standard LLaMA training sbatch"
require_file "$LLAMA_UNLEARN_SBATCH" "LLaMA unlearn training sbatch"
require_file "$QWEN_SBATCH" "standard Qwen training sbatch"
require_file "$QWEN_UNLEARN_SBATCH" "Qwen unlearn training sbatch"
require_file "$BASE_SBATCH" "baseline training sbatch"
require_file "$EVAL_GLOBAL_SBATCH" "global evaluation sbatch"
require_file "$EVAL_CLIENT_SBATCH" "client evaluation sbatch"
require_file "$EVAL_BASE_SBATCH" "baseline evaluation sbatch"
require_file "$EVAL_UNLEARN_GLOBAL_SBATCH" "unlearn global evaluation sbatch"
require_file "$EVAL_UNLEARN_CLIENT_SBATCH" "unlearn client evaluation sbatch"

log "Submitting baseline training (all-in-one scenarios)..."
BASE_JOB=$(submit_sbatch "train-baselines-full" "$BASE_SBATCH")

log "Submitting federated training jobs..."
LLAMA_JOB=$(submit_sbatch "train-llama-full" "$LLAMA_SBATCH")
QWEN_JOB=$(submit_sbatch "train-qwen-full" "$QWEN_SBATCH")
LLAMA_UNLEARN_JOB=$(submit_sbatch "train-llama-unlearn-full" "$LLAMA_UNLEARN_SBATCH")
QWEN_UNLEARN_JOB=$(submit_sbatch "train-qwen-unlearn-full" "$QWEN_UNLEARN_SBATCH")

if [[ $SKIP_GLOBAL_EVAL -eq 0 ]]; then
  log "Submitting global evaluations..."
  submit_sbatch "eval-global-llama" --dependency=afterok:${LLAMA_JOB} --array=0-2 "$EVAL_GLOBAL_SBATCH"
  submit_sbatch "eval-global-qwen" --dependency=afterok:${QWEN_JOB} --array=3-5 "$EVAL_GLOBAL_SBATCH"
  submit_sbatch "eval-global-unlearn-llama" --dependency=afterok:${LLAMA_UNLEARN_JOB} --array=0-5 "$EVAL_UNLEARN_GLOBAL_SBATCH"
  submit_sbatch "eval-global-unlearn-qwen" --dependency=afterok:${QWEN_UNLEARN_JOB} --array=6-11 "$EVAL_UNLEARN_GLOBAL_SBATCH"
else
  log "Skipping global evaluations."
fi

if [[ $SKIP_CLIENT_EVAL -eq 0 ]]; then
  log "Submitting client evaluations..."
  submit_sbatch "eval-clients-llama" --dependency=afterok:${LLAMA_JOB} --array=0-8 "$EVAL_CLIENT_SBATCH"
  submit_sbatch "eval-clients-qwen" --dependency=afterok:${QWEN_JOB} --array=9-17 "$EVAL_CLIENT_SBATCH"
  submit_sbatch "eval-clients-unlearn-llama" --dependency=afterok:${LLAMA_UNLEARN_JOB} --array=0-17 "$EVAL_UNLEARN_CLIENT_SBATCH"
  submit_sbatch "eval-clients-unlearn-qwen" --dependency=afterok:${QWEN_UNLEARN_JOB} --array=18-35 "$EVAL_UNLEARN_CLIENT_SBATCH"
else
  log "Skipping client evaluations."
fi

if [[ $SKIP_BASELINE_EVAL -eq 0 ]]; then
  log "Submitting baseline evaluations..."
  submit_sbatch "eval-baselines-full" --dependency=afterok:${BASE_JOB} "$EVAL_BASE_SBATCH"
else
  log "Skipping baseline evaluations."
fi

log "LLaMA/Qwen pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
