#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
SKIP_PREP=0
GPU_TYPE="H200"
PIPE_ID="${SUPERNI_PIPE_ID:-}"
TRAIN_OPTS="${SUPERNI_TRAIN_OPTS:-}"

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
    --skip-prepare)
      SKIP_PREP=1
      shift
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_superni_pipeline.sh [options]

Options:
  --dry-run       Print the sbatch commands without submitting.
  --skip-eval     Do not launch evaluation array.
  --skip-prepare  Skip the dataset preparation job.
  --gpu-type      GPU type to request (e.g., H200).
  -h, --help      Show this message.
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
  printf "[superni-pipeline] %s\n" "$*" >&2
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
  PIPE_ID="superni_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
LOG_DIR="$ROOT_DIR/superni_logs/$PIPE_ID"
RESULTS_DIR="$ROOT_DIR/superni_results/$PIPE_ID"
run mkdir -p "$LOG_DIR" "$RESULTS_DIR"

SBATCH_EXPORT="ALL,SUPERNI_PIPE_ID=$PIPE_ID,SUPERNI_RESULTS_DIR=$RESULTS_DIR,SUPERNI_LOGS_DIR=$LOG_DIR"

PREP_JOB=""
if [[ $SKIP_PREP -eq 0 ]]; then
  log "Submitting SuperNI data preparation job..."
  PREP_JOB=$(submit_sbatch "prepare-superni-data" \
    --export="$SBATCH_EXPORT" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_prepare_superni_federated.sbatch")
else
  log "Skipping data preparation."
fi

log "Submitting SuperNI training jobs..."
PREP_DEP=()
if [[ -n "$PREP_JOB" ]]; then
  PREP_DEP=("--dependency=afterok:${PREP_JOB}")
fi
FEDAVG_JOB=$(submit_sbatch "train-superni-fedavg" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,SUPERNI_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_superni_fedavg.sbatch")

BANK_JOB=$(submit_sbatch "train-superni-bank" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,SUPERNI_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_superni_bank_perclient.sbatch")

CENTRAL_JOB=$(submit_sbatch "train-superni-centralized" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,SUPERNI_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_superni_centralized.sbatch")

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Submitting SuperNI evaluations..."
  EVAL_JOB=$(submit_sbatch "eval-superni" \
    --dependency=afterok:${FEDAVG_JOB}:${BANK_JOB}:${CENTRAL_JOB} \
    --array=0-8 \
    --export="$SBATCH_EXPORT" \
    --gres="gpu:${GPU_TYPE}:1" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --error="$LOG_DIR/%x_%A_%a.err" \
    "$SBATCH_DIR/sbatch_eval_superni_global.sbatch")

  submit_sbatch "collect-superni-fedavg" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,SUPERNI_EXP=fedavg,SUPERNI_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_superni_metrics.sbatch"
  submit_sbatch "collect-superni-bank" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,SUPERNI_EXP=bank_perclient,SUPERNI_EVAL_JOB_ID=$EVAL_JOB,SUPERNI_FEDAVG_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_superni_metrics.sbatch"
  submit_sbatch "collect-superni-centralized" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,SUPERNI_EXP=centralized,SUPERNI_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_superni_metrics.sbatch"
else
  log "Skipping evaluations."
fi

log "SuperNI pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
