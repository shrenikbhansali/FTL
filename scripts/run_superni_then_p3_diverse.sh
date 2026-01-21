#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
OVERLAP_PREP=1
SKIP_WARMUP=0
SKIP_SUPERNI_PREP=0
NO_CONDA=0
GPUS="0,1,2,3"
PIPE_ID="${PIPE_ID:-}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""
SUPERNI_OPTS=""
P3_OPTS=""

usage() {
  cat <<'USAGE'
Usage: run_superni_then_p3_diverse.sh [options]

Options:
  --dry-run            Print commands without running them.
  --resume             Reuse existing outputs when possible.
  --no-overlap-prep    Do not overlap P3 data prep with SuperNI training.
  --skip-warmup        Skip eval-data warmup before P3 eval.
  --skip-superni-prep  Skip SuperNI data preparation step.
  --no-conda           Do not attempt to activate conda env.
  --conda-env          Conda env path to activate.
  --hf-home            Override HF_HOME for dataset/model cache.
  --gpus               Comma-separated GPU list (default: 0,1,2,3).
  --pipe-id            Override wrapper pipeline id (default: auto).
  --superni-opts        Extra options for run_local_superni_pipeline.sh
                       (use :: as separator).
  --p3-opts            Extra options for run_local_p3_diverse_pipeline.sh
                       (use :: as separator).
  -h, --help           Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-overlap-prep)
      OVERLAP_PREP=0
      shift
      ;;
    --skip-warmup)
      SKIP_WARMUP=1
      shift
      ;;
    --skip-superni-prep)
      SKIP_SUPERNI_PREP=1
      shift
      ;;
    --no-conda)
      NO_CONDA=1
      shift
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --hf-home)
      HF_HOME_OVERRIDE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --pipe-id)
      PIPE_ID="$2"
      shift 2
      ;;
    --superni-opts)
      SUPERNI_OPTS="$2"
      shift 2
      ;;
    --p3-opts)
      P3_OPTS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="superni_then_p3_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

LOG_DIR="$ROOT_DIR/local_pipeline_logs/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
PREP_LOG="$LOG_DIR/p3_prep.log"
SUPERNI_PREP_LOG="$LOG_DIR/superni_prep.log"
mkdir -p "$LOG_DIR"
: > "$PIPE_LOG"

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  printf "[%s] %s\n" "$(ts)" "$*" | tee -a "$PIPE_LOG" >&2
}

maybe_activate_conda() {
  local env_name="$1"
  if [[ -z "$env_name" ]]; then
    return 0
  fi
  if [[ "${CONDA_PREFIX:-}" == "$env_name" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    return 0
  fi
  local conda_bin="/nethome/sbhansali8/miniconda3/bin/conda"
  if [[ -x "$conda_bin" ]]; then
    # shellcheck disable=SC1090
    eval "$("$conda_bin" shell.bash hook)"
    conda activate "$env_name"
    return 0
  fi
  log "conda not found; continuing without activation."
}

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi

if [[ $NO_CONDA -eq 0 ]]; then
  maybe_activate_conda "$CONDA_ENV"
fi

SUPERNI_CMD=(scripts/run_local_superni_pipeline.sh --gpus "$GPUS")
P3_CMD=(scripts/run_local_p3_diverse_pipeline.sh --gpus "$GPUS")
PREP_CMD=(scripts/run_local_p3_diverse_pipeline.sh --prep-only --gpus "$GPUS")
SUPERNI_PREP_CMD=(scripts/run_local_prepare_data.sh --skip-p3)

if [[ $RESUME -eq 1 ]]; then
  SUPERNI_CMD+=(--resume)
  P3_CMD+=(--resume)
  PREP_CMD+=(--resume)
else
  SUPERNI_PREP_CMD+=(--overwrite)
fi

if [[ -n "$CONDA_ENV" ]]; then
  SUPERNI_CMD+=(--conda-env "$CONDA_ENV")
  P3_CMD+=(--conda-env "$CONDA_ENV")
  PREP_CMD+=(--conda-env "$CONDA_ENV")
  SUPERNI_PREP_CMD+=(--conda-env "$CONDA_ENV")
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  SUPERNI_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
  P3_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
  PREP_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
  SUPERNI_PREP_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi

if [[ -n "$SUPERNI_OPTS" ]]; then
  read -r -a SUPERNI_EXTRA <<< "${SUPERNI_OPTS//::/ }"
  SUPERNI_CMD+=("${SUPERNI_EXTRA[@]}")
fi
if [[ -n "$P3_OPTS" ]]; then
  read -r -a P3_EXTRA <<< "${P3_OPTS//::/ }"
  P3_CMD+=("${P3_EXTRA[@]}")
  PREP_CMD+=("${P3_EXTRA[@]}")
fi

if [[ $DRY_RUN -eq 1 ]]; then
  if [[ $SKIP_SUPERNI_PREP -eq 0 ]]; then
    log "DRY-RUN: ${SUPERNI_PREP_CMD[*]}"
  fi
  log "DRY-RUN: ${SUPERNI_CMD[*]}"
  if [[ $OVERLAP_PREP -eq 1 ]]; then
    log "DRY-RUN (overlap prep): ${PREP_CMD[*]}"
  fi
  log "DRY-RUN: ${P3_CMD[*]}"
  exit 0
fi

if [[ $SKIP_SUPERNI_PREP -eq 0 ]]; then
  log "Starting SuperNI data prep (log: $SUPERNI_PREP_LOG)."
  "${SUPERNI_PREP_CMD[@]}" >"$SUPERNI_PREP_LOG" 2>&1
  log "SuperNI data prep complete."
else
  log "Skipping SuperNI data prep."
fi

PREP_PID=""
if [[ $OVERLAP_PREP -eq 1 ]]; then
  log "Starting P3 diverse prep in background (log: $PREP_LOG)."
  "${PREP_CMD[@]}" >"$PREP_LOG" 2>&1 &
  PREP_PID=$!
fi

log "Starting SuperNI pipeline."
"${SUPERNI_CMD[@]}"
log "SuperNI pipeline complete."

if [[ -n "$PREP_PID" ]]; then
  log "Waiting for P3 prep (pid=$PREP_PID)."
  if ! wait "$PREP_PID"; then
    log "P3 prep failed; see $PREP_LOG."
    exit 1
  fi
  log "P3 prep complete."
fi

if [[ $SKIP_WARMUP -eq 0 ]]; then
  log "Running eval-data warmup for P3."
  bash scripts/warmup_eval_data.sh
fi

log "Starting P3 diverse pipeline."
"${P3_CMD[@]}"
log "P3 diverse pipeline complete."
