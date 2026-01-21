#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
SKIP_EVAL=0
NO_CONDA=0
PREP_ONLY=0
GPUS="0,1,2,3"
PIPE_ID="${P3_DIVERSE_PIPE_ID:-}"
CONDA_ENV="/home/heck2/sbhansali8/condastuff/fs-llm"
HF_HOME_OVERRIDE=""

CONFIG_OUTPUT="materials/p3_config_diverse.txt"
CONFIG_DATASET="bigscience/P3"
CONFIGS_PER_PREFIX=1
CONFIG_SHUFFLE=0

DATA_OUTPUT="data/p3_federated_config_diverse"
MAX_TOTAL_SAMPLES=250000
MAX_SAMPLES_PER_CLIENT=2000
MAX_CLIENTS="160"
DROP_LONG=0

TRAIN_OPTS="${P3_TRAIN_OPTS:-}"

usage() {
  cat <<'USAGE'
Usage: run_local_p3_diverse_pipeline.sh [options]

Options:
  --dry-run               Print commands without running them.
  --resume                Skip steps with existing outputs.
  --skip-eval              Skip evaluations.
  --prep-only             Only build config list and prepare data (skip train/eval).
  --no-conda              Do not attempt to activate conda env.
  --conda-env             Conda env path to activate.
  --hf-home               Override HF_HOME for cache.
  --gpus                  Comma-separated GPU list (default: 0,1,2,3).
  --pipe-id               Override pipeline id.

  --config-output         Where to write diverse config list.
  --config-dataset        HF dataset id (default: bigscience/P3).
  --configs-per-prefix    Number of configs per dataset prefix.
  --config-shuffle        Shuffle configs before selection.

  --data-output           Output dir for diverse config split.
  --max-total-samples     Cap total written samples (default: 250000).
  --max-samples-per-client Cap per-client samples (default: 2000).
  --max-clients           Limit number of clients (default: 160).
  --drop-long             Drop long samples with tokenizer length filter.
  --no-drop-long          Keep long samples (disable length filter; default).

  --train-opts            Extra federatedscope overrides (use :: as separator).
  -h, --help              Show this message.
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
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --prep-only)
      PREP_ONLY=1
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
    --config-output)
      CONFIG_OUTPUT="$2"
      shift 2
      ;;
    --config-dataset)
      CONFIG_DATASET="$2"
      shift 2
      ;;
    --configs-per-prefix)
      CONFIGS_PER_PREFIX="$2"
      shift 2
      ;;
    --config-shuffle)
      CONFIG_SHUFFLE=1
      shift
      ;;
    --data-output)
      DATA_OUTPUT="$2"
      shift 2
      ;;
    --max-total-samples)
      MAX_TOTAL_SAMPLES="$2"
      shift 2
      ;;
    --max-samples-per-client)
      MAX_SAMPLES_PER_CLIENT="$2"
      shift 2
      ;;
    --max-clients)
      MAX_CLIENTS="$2"
      shift 2
      ;;
    --drop-long)
      DROP_LONG=1
      shift
      ;;
    --no-drop-long)
      DROP_LONG=0
      shift
      ;;
    --train-opts)
      TRAIN_OPTS="$2"
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
  PIPE_ID="p3_diverse_$(date +%Y%m%d_%H%M%S)_$RANDOM"
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

CONFIG_CMD=(python scripts/build_p3_diverse_config_list.py
  --dataset "$CONFIG_DATASET"
  --output "$CONFIG_OUTPUT"
  --max-per-prefix "$CONFIGS_PER_PREFIX"
)
if [[ $CONFIG_SHUFFLE -eq 1 ]]; then
  CONFIG_CMD+=(--shuffle)
fi
if [[ $RESUME -eq 1 && -f "$CONFIG_OUTPUT" ]]; then
  log "Skipping build-config-list; config list exists at $CONFIG_OUTPUT."
else
  run_stage "build-config-list" "${CONFIG_CMD[@]}"
fi

PREP_CMD=(python scripts/prepare_p3_federated.py
  --output-dir "$DATA_OUTPUT"
  --group-by config
  --config-file "$CONFIG_OUTPUT"
  --max-total-samples "$MAX_TOTAL_SAMPLES"
  --max-samples-per-client "$MAX_SAMPLES_PER_CLIENT"
  --val-fraction 0.01
  --seed 42
  --streaming
)
if [[ -n "$MAX_CLIENTS" ]]; then
  PREP_CMD+=(--max-clients "$MAX_CLIENTS")
fi
if [[ $DROP_LONG -eq 0 ]]; then
  PREP_CMD+=(--no-drop-long)
fi
if [[ $RESUME -eq 1 && -f "$DATA_OUTPUT/manifest.json" ]]; then
  log "Skipping prepare-data; manifest exists at $DATA_OUTPUT."
else
  PREP_CMD+=(--overwrite)
  run_stage "prepare-data" "${PREP_CMD[@]}"
fi

if [[ $PREP_ONLY -eq 1 ]]; then
  log "Prep-only requested; skipping train/eval."
  exit 0
fi

PIPELINE_CMD=(scripts/run_local_p3_pipeline.sh
  --variant config
  --gpus "$GPUS"
  --pipe-id "$PIPE_ID"
  --train-opts "data.tulu3_federated.root ${DATA_OUTPUT#data/}::${TRAIN_OPTS}"
)
if [[ $DRY_RUN -eq 1 ]]; then
  PIPELINE_CMD+=(--dry-run)
fi
if [[ $RESUME -eq 1 ]]; then
  PIPELINE_CMD+=(--resume)
fi
if [[ $SKIP_EVAL -eq 1 ]]; then
  PIPELINE_CMD+=(--skip-eval)
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  PIPELINE_CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi

run_stage "train-eval" "${PIPELINE_CMD[@]}"

log "P3 diverse pipeline complete."
