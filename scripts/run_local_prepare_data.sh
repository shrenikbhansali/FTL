#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OVERWRITE=0
SKIP_SUPERNI=0
SKIP_P3=0
P3_VARIANTS="dataset"
SUPERNI_OUTPUT="data/superni_federated"
SUPERNI_MAX_TOTAL_SAMPLES="${SUPERNI_MAX_TOTAL_SAMPLES:-400000}"
SUPERNI_MAX_SAMPLES_PER_CLIENT="${SUPERNI_MAX_SAMPLES_PER_CLIENT:-50000}"
P3_OUTPUT_BASE="data/p3_federated"
P3_CATEGORY_MAP_OVERRIDE=""
CONDA_ENV=""
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_prepare_data.sh [options]

Options:
  --dry-run            Print commands without running them.
  --overwrite          Remove existing output directories before writing.
  --skip-superni       Skip SuperNI data preparation.
  --skip-p3            Skip P3 data preparation.
  --p3-variants        Comma-separated list: config,dataset,category or "all".
  --superni-output     Output directory for SuperNI (default: data/superni_federated).
  --p3-output-base     Output base for P3 (default: data/p3_federated).
  --p3-category-map    Category map JSON for P3 grouping by category.
  --conda-env          Conda env to activate before running.
  --hf-home            Override HF_HOME for dataset cache.
  -h, --help           Show this message.

Notes:
  - Extra P3 options can be provided via env vars:
    P3_CONFIG_FILE, P3_CONFIGS, P3_CATEGORY_MAP, P3_MAX_CLIENTS.
  - Extra SuperNI options via env vars:
    SUPERNI_MAX_CLIENTS, SUPERNI_MAX_TOTAL_SAMPLES, SUPERNI_MAX_SAMPLES_PER_CLIENT.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --skip-superni)
      SKIP_SUPERNI=1
      shift
      ;;
    --skip-p3)
      SKIP_P3=1
      shift
      ;;
    --p3-variants)
      P3_VARIANTS="$2"
      shift 2
      ;;
    --superni-output)
      SUPERNI_OUTPUT="$2"
      shift 2
      ;;
    --p3-output-base)
      P3_OUTPUT_BASE="$2"
      shift 2
      ;;
    --p3-category-map)
      P3_CATEGORY_MAP_OVERRIDE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --hf-home)
      HF_HOME_OVERRIDE="$2"
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

PIPE_ID="${DATA_PIPE_ID:-}"
if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="data_prep_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

LOG_DIR="$ROOT_DIR/data_prep_logs/$PIPE_ID"
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

maybe_activate_conda "$CONDA_ENV"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export PYTHONUNBUFFERED=1

P3_VARIANTS="${P3_VARIANTS// /}"
P3_VARIANTS="${P3_VARIANTS,,}"
if [[ "$P3_VARIANTS" == "all" ]]; then
  P3_VARIANTS="config,dataset,category"
fi
IFS=',' read -r -a P3_VARIANT_LIST <<< "$P3_VARIANTS"
if [[ ${#P3_VARIANT_LIST[@]} -eq 0 ]]; then
  log "No P3 variants provided."
  exit 1
fi
for variant in "${P3_VARIANT_LIST[@]}"; do
  case "$variant" in
    config|dataset|category)
      ;;
    *)
      log "Unknown P3 variant: $variant"
      exit 1
      ;;
  esac
done

run_cmd() {
  local label="$1"
  local log_file="$2"
  shift 2
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN [$label]: $*"
    return 0
  fi
  : > "$log_file"
  log "Running [$label] (log: $log_file)"
  "$@" >> "$log_file" 2>&1
}

should_skip_dir() {
  local out_dir="$1"
  if [[ $OVERWRITE -eq 0 && -f "$out_dir/manifest.json" ]]; then
    return 0
  fi
  return 1
}

if [[ $SKIP_SUPERNI -eq 0 ]]; then
  SUPERNI_DIR="$SUPERNI_OUTPUT"
  if should_skip_dir "$SUPERNI_DIR"; then
    log "Skipping SuperNI prep; manifest exists at $SUPERNI_DIR."
  else
    SUPERNI_LOG="$LOG_DIR/prepare_superni.log"
    SUPERNI_CMD=(python scripts/prepare_superni_federated.py
      --dataset "Muennighoff/natural-instructions"
      --output-dir "$SUPERNI_DIR"
      --group-by task_name
      --dedupe-by id
      --max-length 2048
      --max-total-samples "$SUPERNI_MAX_TOTAL_SAMPLES"
      --max-samples-per-client "$SUPERNI_MAX_SAMPLES_PER_CLIENT"
      --val-fraction 0.01
      --seed 42
      --streaming
    )
    if [[ -n "${SUPERNI_MAX_CLIENTS:-}" ]]; then
      SUPERNI_CMD+=(--max-clients "$SUPERNI_MAX_CLIENTS")
    fi
    if [[ $OVERWRITE -eq 1 ]]; then
      SUPERNI_CMD+=(--overwrite)
    fi
    run_cmd "prepare-superni" "$SUPERNI_LOG" "${SUPERNI_CMD[@]}"
  fi
else
  log "Skipping SuperNI prep (per flag)."
fi

if [[ $SKIP_P3 -eq 0 ]]; then
  CATEGORY_MAP_DEFAULT="$ROOT_DIR/materials/p3_category_map.json"
  for variant in "${P3_VARIANT_LIST[@]}"; do
    P3_DIR="${P3_OUTPUT_BASE}_${variant}"
    if should_skip_dir "$P3_DIR"; then
      log "Skipping P3 $variant prep; manifest exists at $P3_DIR."
      continue
    fi
    P3_LOG="$LOG_DIR/prepare_p3_${variant}.log"
    P3_CMD=(python scripts/prepare_p3_federated.py
      --output-dir "$P3_DIR"
      --group-by "$variant"
      --max-length 2048
      --max-total-samples 200000
      --max-samples-per-client 20000
      --val-fraction 0.01
      --seed 42
      --streaming
    )
    if [[ -n "${P3_CONFIG_FILE:-}" ]]; then
      P3_CMD+=(--config-file "$P3_CONFIG_FILE")
    fi
    if [[ -n "${P3_CONFIGS:-}" ]]; then
      P3_CMD+=(--configs "$P3_CONFIGS")
    fi
    if [[ -n "${P3_MAX_CLIENTS:-}" ]]; then
      P3_CMD+=(--max-clients "$P3_MAX_CLIENTS")
    fi
    if [[ "$variant" == "category" ]]; then
      CATEGORY_MAP="${P3_CATEGORY_MAP_OVERRIDE:-${P3_CATEGORY_MAP:-$CATEGORY_MAP_DEFAULT}}"
      if [[ ! -f "$CATEGORY_MAP" ]]; then
        log "P3 category map not found at $CATEGORY_MAP; aborting category prep."
        exit 1
      fi
      P3_CMD+=(--category-map "$CATEGORY_MAP")
    fi
    if [[ $OVERWRITE -eq 1 ]]; then
      P3_CMD+=(--overwrite)
    fi
    run_cmd "prepare-p3-$variant" "$P3_LOG" "${P3_CMD[@]}"
  done
else
  log "Skipping P3 prep (per flag)."
fi

log "Data preparation complete."
