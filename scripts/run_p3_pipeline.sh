#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
SKIP_PREP=0
GPU_TYPE="H200"
PIPE_ID="${P3_PIPE_ID:-}"
TRAIN_OPTS="${P3_TRAIN_OPTS:-}"
VARIANT="dataset"

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
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_p3_pipeline.sh [options]

Options:
  --dry-run       Print the sbatch commands without submitting.
  --skip-eval     Do not launch evaluation array.
  --skip-prepare  Skip the dataset preparation job.
  --gpu-type      GPU type to request (e.g., H200).
  --variant       One of: config, dataset, category.
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

case "$VARIANT" in
  config|dataset|category)
    ;;
  *)
    echo "Unknown variant: $VARIANT (expected: config, dataset, category)" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SBATCH_DIR="$ROOT_DIR/scripts"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Run this script on a Slurm login node." >&2
  exit 1
fi

log() {
  printf "[p3-pipeline] %s\n" "$*" >&2
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
  PIPE_ID="p3_${VARIANT}_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
LOG_DIR="$ROOT_DIR/p3_logs/$VARIANT/$PIPE_ID"
RESULTS_DIR="$ROOT_DIR/p3_results/$VARIANT/$PIPE_ID"
run mkdir -p "$LOG_DIR" "$RESULTS_DIR"

DATA_ROOT="data/p3_federated_${VARIANT}"
GROUP_BY="$VARIANT"
CATEGORY_MAP="$ROOT_DIR/materials/p3_category_map.json"

BASE_TRAIN_OPTS="data.tulu3_federated.root ${DATA_ROOT#data/}"
if [[ -n "$TRAIN_OPTS" ]]; then
  TRAIN_OPTS="${BASE_TRAIN_OPTS}::${TRAIN_OPTS}"
else
  TRAIN_OPTS="$BASE_TRAIN_OPTS"
fi

SBATCH_EXPORT="ALL,P3_PIPE_ID=$PIPE_ID,P3_RESULTS_DIR=$RESULTS_DIR,P3_LOGS_DIR=$LOG_DIR,P3_DATA_ROOT=$DATA_ROOT,P3_GROUP_BY=$GROUP_BY,P3_CATEGORY_MAP=$CATEGORY_MAP"

PREP_JOB=""
if [[ $SKIP_PREP -eq 0 ]]; then
  log "Submitting P3 data preparation job..."
  PREP_JOB=$(submit_sbatch "prepare-p3-data" \
    --export="$SBATCH_EXPORT" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_prepare_p3_federated.sbatch")
else
  log "Skipping data preparation."
fi

log "Submitting P3 training jobs..."
PREP_DEP=()
if [[ -n "$PREP_JOB" ]]; then
  PREP_DEP=("--dependency=afterok:${PREP_JOB}")
fi
FEDAVG_JOB=$(submit_sbatch "train-p3-fedavg" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,P3_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_p3_fedavg.sbatch")

BANK_JOB=$(submit_sbatch "train-p3-bank" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,P3_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_p3_bank_perclient.sbatch")

CENTRAL_JOB=$(submit_sbatch "train-p3-centralized" \
  "${PREP_DEP[@]}" \
  --export="$SBATCH_EXPORT,P3_TRAIN_OPTS=$TRAIN_OPTS" \
  --gres="gpu:${GPU_TYPE}:1" \
  --output="$LOG_DIR/%x_%A.out" \
  --error="$LOG_DIR/%x_%A.err" \
  "$SBATCH_DIR/sbatch_train_p3_centralized.sbatch")

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Submitting P3 evaluations..."
  EVAL_JOB=$(submit_sbatch "eval-p3" \
    --dependency=afterok:${FEDAVG_JOB}:${BANK_JOB}:${CENTRAL_JOB} \
    --array=0-8 \
    --export="$SBATCH_EXPORT" \
    --gres="gpu:${GPU_TYPE}:1" \
    --output="$LOG_DIR/%x_%A_%a.out" \
    --error="$LOG_DIR/%x_%A_%a.err" \
    "$SBATCH_DIR/sbatch_eval_p3_global.sbatch")

  submit_sbatch "collect-p3-fedavg" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,P3_EXP=fedavg,P3_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_p3_metrics.sbatch"
  submit_sbatch "collect-p3-bank" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,P3_EXP=bank_perclient,P3_EVAL_JOB_ID=$EVAL_JOB,P3_FEDAVG_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_p3_metrics.sbatch"
  submit_sbatch "collect-p3-centralized" --dependency=afterok:${EVAL_JOB} \
    --export="$SBATCH_EXPORT,P3_EXP=centralized,P3_EVAL_JOB_ID=$EVAL_JOB" \
    --output="$LOG_DIR/%x_%A.out" \
    --error="$LOG_DIR/%x_%A.err" \
    "$SBATCH_DIR/sbatch_collect_p3_metrics.sbatch"
else
  log "Skipping evaluations."
fi

log "P3 pipeline submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
