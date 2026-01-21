#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
SKIP_DATA=0
SKIP_P3=0
SKIP_SUPERNI=0
SKIP_EVAL=0
OVERWRITE_DATA=0
NO_CONDA=0
GPUS="0,1,2,3"
P3_VARIANTS="all"
P3_CATEGORY_MAP=""
P3_TRAIN_OPTS=""
SUPERNI_TRAIN_OPTS=""
PIPE_ID="${LOCAL_PIPE_ID:-}"
CONDA_ENV="/home/heck2/sbhansali8/condastuff/fs-llm"
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_full_pipeline.sh [options]

Options:
  --dry-run          Print commands without running them.
  --resume           Skip steps with existing outputs.
  --skip-data        Skip data preparation stage.
  --skip-p3          Skip P3 pipeline stage.
  --skip-superni     Skip SuperNI pipeline stage.
  --skip-eval        Skip evaluations in P3 and SuperNI pipelines.
  --overwrite-data   Regenerate datasets (removes existing outputs).
  --no-conda         Do not attempt to activate conda env.
  --conda-env        Conda env path to activate.
  --hf-home          Override HF_HOME for cache (default: /home/heck2/sbhansali8/HFcache).
  --gpus             Comma-separated GPU list (default: 0,1,2,3).
  --p3-variants      Comma-separated list or "all" (default: all).
  --p3-category-map  Category map JSON for P3 grouping by category.
  --p3-train-opts    Extra P3 train opts (use :: as separator).
  --superni-train-opts Extra SuperNI train opts (use :: as separator).
  --pipe-id          Override top-level pipeline id.
  -h, --help         Show this message.
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
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --skip-p3)
      SKIP_P3=1
      shift
      ;;
    --skip-superni)
      SKIP_SUPERNI=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --overwrite-data)
      OVERWRITE_DATA=1
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
    --p3-variants)
      P3_VARIANTS="$2"
      shift 2
      ;;
    --p3-category-map)
      P3_CATEGORY_MAP="$2"
      shift 2
      ;;
    --p3-train-opts)
      P3_TRAIN_OPTS="$2"
      shift 2
      ;;
    --superni-train-opts)
      SUPERNI_TRAIN_OPTS="$2"
      shift 2
      ;;
    --pipe-id)
      PIPE_ID="$2"
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
  PIPE_ID="local_full_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

LOG_DIR="$ROOT_DIR/local_pipeline_logs/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
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

run_stage() {
  local label="$1"
  shift
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN [$label]: $*"
    return 0
  fi
  log "Stage start: $label"
  "$@"
  log "Stage done: $label"
}

DATA_CMD=(scripts/run_local_prepare_data.sh --p3-variants "$P3_VARIANTS")
if [[ $DRY_RUN -eq 1 ]]; then
  DATA_CMD+=(--dry-run)
fi
if [[ $OVERWRITE_DATA -eq 1 ]]; then
  DATA_CMD+=(--overwrite)
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  DATA_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi
if [[ -n "$P3_CATEGORY_MAP" ]]; then
  DATA_CMD+=(--p3-category-map "$P3_CATEGORY_MAP")
fi
if [[ $SKIP_P3 -eq 1 ]]; then
  DATA_CMD+=(--skip-p3)
fi
if [[ $SKIP_SUPERNI -eq 1 ]]; then
  DATA_CMD+=(--skip-superni)
fi
if [[ $SKIP_DATA -eq 0 ]]; then
  run_stage "data-prep" "${DATA_CMD[@]}"
else
  log "Skipping data preparation stage."
fi

P3_CMD=(scripts/run_local_p3_pipeline.sh --variants "$P3_VARIANTS" --gpus "$GPUS")
if [[ $DRY_RUN -eq 1 ]]; then
  P3_CMD+=(--dry-run)
fi
if [[ $RESUME -eq 1 ]]; then
  P3_CMD+=(--resume)
fi
if [[ $SKIP_EVAL -eq 1 ]]; then
  P3_CMD+=(--skip-eval)
fi
if [[ -n "$P3_TRAIN_OPTS" ]]; then
  P3_CMD+=(--train-opts "$P3_TRAIN_OPTS")
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  P3_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi
if [[ $SKIP_P3 -eq 0 ]]; then
  run_stage "p3-pipeline" "${P3_CMD[@]}"
else
  log "Skipping P3 pipeline stage."
fi

SUPERNI_CMD=(scripts/run_local_superni_pipeline.sh --gpus "$GPUS")
if [[ $DRY_RUN -eq 1 ]]; then
  SUPERNI_CMD+=(--dry-run)
fi
if [[ $RESUME -eq 1 ]]; then
  SUPERNI_CMD+=(--resume)
fi
if [[ $SKIP_EVAL -eq 1 ]]; then
  SUPERNI_CMD+=(--skip-eval)
fi
if [[ -n "$SUPERNI_TRAIN_OPTS" ]]; then
  SUPERNI_CMD+=(--train-opts "$SUPERNI_TRAIN_OPTS")
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  SUPERNI_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi
if [[ $SKIP_SUPERNI -eq 0 ]]; then
  run_stage "superni-pipeline" "${SUPERNI_CMD[@]}"
else
  log "Skipping SuperNI pipeline stage."
fi

log "Local full pipeline complete."
