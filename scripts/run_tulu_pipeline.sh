#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
GPU_TYPE="H200"
PIPE_ID="${TULU_PIPE_ID:-}"
COMMON_OPTS="${TULU_TRAIN_OPTS_COMMON:-}"
BANK_OPTS="${TULU_TRAIN_OPTS_BANK:-}"
OPTS_DELIM="::"

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
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_tulu_pipeline.sh [options]

Options:
  --dry-run   Print the sbatch commands without submitting.
  --skip-eval Do not launch evaluation array.
  --gpu-type  GPU type to request (e.g., H100, H200).
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

gen_run_id() {
  python - <<'PY'
import uuid
print(uuid.uuid4().hex)
PY
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

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
LOG_DIR="$ROOT_DIR/tulupipe_logs/$PIPE_ID"
RESULTS_DIR="$ROOT_DIR/tulupipe_results/$PIPE_ID"
run mkdir -p "$LOG_DIR" "$RESULTS_DIR"

FEDAVG_WANDB_RUN_ID="${WANDB_RUN_ID_FEDAVG:-$(gen_run_id)}"
BANK_WANDB_RUN_ID="${WANDB_RUN_ID_BANK:-$(gen_run_id)}"

SBATCH_EXPORT="ALL,TULU_PIPE_ID=$PIPE_ID,TULU_RESULTS_DIR=$RESULTS_DIR,TULU_LOGS_DIR=$LOG_DIR"
BANK_TRAIN_OPTS="$COMMON_OPTS"
if [[ -n "$BANK_OPTS" ]]; then
  if [[ -n "$BANK_TRAIN_OPTS" ]]; then
    BANK_TRAIN_OPTS="${BANK_TRAIN_OPTS}${OPTS_DELIM}${BANK_OPTS}"
  else
    BANK_TRAIN_OPTS="$BANK_OPTS"
  fi
fi
log "Submitting Tulu training jobs..."
FEDAVG_JOB=$(submit_sbatch "train-tulu-fedavg" \
  --export="$SBATCH_EXPORT,WANDB_RUN_ID=$FEDAVG_WANDB_RUN_ID,WANDB_RESUME=allow,TULU_TRAIN_OPTS=$COMMON_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_tulu_fedavg.sbatch")
BANK_JOB=$(submit_sbatch "train-tulu-bank-perclient" \
  --export="$SBATCH_EXPORT,WANDB_RUN_ID=$BANK_WANDB_RUN_ID,WANDB_RESUME=allow,TULU_TRAIN_OPTS=$BANK_TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_tulu_bank_perclient.sbatch")

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Submitting Tulu evaluations..."
  FEDAVG_EVAL_JOB=$(submit_sbatch "eval-tulu-fedavg" --dependency=afterok:${FEDAVG_JOB} --array=0-2 \
    --export="$SBATCH_EXPORT" \
    --gres="gpu:${GPU_TYPE}:1" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --error="$LOG_DIR/%x_%A_%a.err" \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch")
  BANK_EVAL_JOB=$(submit_sbatch "eval-tulu-bank" --dependency=afterok:${BANK_JOB} --array=3-5 \
    --export="$SBATCH_EXPORT" \
    --gres="gpu:${GPU_TYPE}:1" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --error="$LOG_DIR/%x_%A_%a.err" \
    "$SBATCH_DIR/sbatch_eval_tulu_global.sbatch")

  submit_sbatch "collect-tulu-fedavg" --dependency=afterok:${FEDAVG_EVAL_JOB} \
    --export="$SBATCH_EXPORT,TULU_EXP=fedavg,TULU_EVAL_JOB_ID=$FEDAVG_EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_tulu_metrics.sbatch"
  submit_sbatch "collect-tulu-bank" --dependency=afterok:${BANK_EVAL_JOB} \
    --export="$SBATCH_EXPORT,TULU_EXP=bank_perclient,TULU_EVAL_JOB_ID=$BANK_EVAL_JOB,TULU_FEDAVG_EVAL_JOB_ID=$FEDAVG_EVAL_JOB,WANDB_RUN_ID=$BANK_WANDB_RUN_ID,WANDB_RESUME=allow" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_tulu_metrics.sbatch"
else
  log "Skipping evaluations."
fi

log "Tulu pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
