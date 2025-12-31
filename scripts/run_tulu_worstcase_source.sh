#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
GPU_TYPE="H200"
PIPE_ID="${TULU_PIPE_ID:-}"
TRAIN_OPTS="${TULU_TRAIN_OPTS:-}"

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
Usage: run_tulu_worstcase_source.sh [options]

Options:
  --dry-run   Print the sbatch commands without submitting.
  --skip-eval Do not launch evaluation array.
  --gpu-type  GPU type to request (e.g., H200).
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
  printf "[tulu-worstcase] %s\n" "$*" >&2
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
  PIPE_ID="wc_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
LOG_DIR="$ROOT_DIR/tulu_worstcase_logs/$PIPE_ID"
RESULTS_DIR="$ROOT_DIR/tulu_worstcase_results/$PIPE_ID"
run mkdir -p "$LOG_DIR" "$RESULTS_DIR"

SBATCH_EXPORT="ALL,TULU_PIPE_ID=$PIPE_ID,TULU_RESULTS_DIR=$RESULTS_DIR,TULU_LOGS_DIR=$LOG_DIR"

log "Submitting worst-case Tulu training jobs..."
FEDAVG_JOB=$(submit_sbatch "train-worstcase-fedavg" \
  --export="$SBATCH_EXPORT,TULU_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_tulu_worstcase_fedavg.sbatch")
BANK_JOB=$(submit_sbatch "train-worstcase-bank" \
  --export="$SBATCH_EXPORT,TULU_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_tulu_worstcase_bank_perclient.sbatch")
CENTRAL_JOB=$(submit_sbatch "train-worstcase-centralized" \
  --export="$SBATCH_EXPORT,TULU_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_tulu_worstcase_centralized.sbatch")

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Submitting worst-case Tulu evaluations..."
  EVAL_JOB=$(submit_sbatch "eval-tulu-worstcase" \
    --dependency=afterok:${FEDAVG_JOB}:${BANK_JOB}:${CENTRAL_JOB} \
    --array=0-8 \
    --export="$SBATCH_EXPORT" \
    --gres="gpu:${GPU_TYPE}:1" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --error="$LOG_DIR/%x_%A_%a.err" \
    "$SBATCH_DIR/sbatch_eval_tulu_global_worstcase.sbatch")

  submit_sbatch "collect-worstcase-fedavg" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,TULU_EXP=fedavg,TULU_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_tulu_metrics.sbatch"
  submit_sbatch "collect-worstcase-bank" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,TULU_EXP=bank_perclient,TULU_EVAL_JOB_ID=$EVAL_JOB,TULU_FEDAVG_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_tulu_metrics.sbatch"
  submit_sbatch "collect-worstcase-centralized" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,TULU_EXP=centralized,TULU_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_tulu_metrics.sbatch"
else
  log "Skipping evaluations."
fi

log "Worst-case Tulu pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
