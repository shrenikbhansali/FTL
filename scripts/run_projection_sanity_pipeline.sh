#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_projection_sanity_pipeline.sh [options]

Run two short (10-round) projection sanity trainings:
  - proj_on (projection enabled)
  - proj_off (projection disabled)

Options:
  --gpus LIST          Comma-separated GPU IDs (default: 0,1)
  --pipe-id ID         Override pipeline id (default: auto)
  --resume             Skip jobs with existing outputs
  --hf-home PATH       Override HF_HOME
  -h, --help           Show this help
USAGE
}

GPU_LIST="0,1"
PIPE_ID=""
RESUME=0
HF_HOME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --pipe-id)
      PIPE_ID="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
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

CFG_DIR="$ROOT_DIR/final/configs/projection_sanity"
CFG_ON="$CFG_DIR/individual_federated_bank_perclient_proj_on_10r.yaml"
CFG_OFF="$CFG_DIR/individual_federated_bank_perclient_proj_off_10r.yaml"

if [[ ! -f "$CFG_ON" || ! -f "$CFG_OFF" ]]; then
  echo "Missing configs under $CFG_DIR" >&2
  exit 1
fi

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="projection_sanity_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

RESULTS_ROOT="$ROOT_DIR/final/results/projection_sanity/$PIPE_ID"
LOG_ROOT="$ROOT_DIR/final/logs/projection_sanity/$PIPE_ID"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi

ensure_conda_libs() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    local conda_lib="$CONDA_PREFIX/lib"
    if [[ -d "$conda_lib" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$conda_lib:"*) ;;
        *) export LD_LIBRARY_PATH="$conda_lib:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  fi
}
ensure_conda_libs

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$msg" | tee -a "$LOG_ROOT/pipeline.log"
}

run_job() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  LAST_PID=""

  local outdir="$RESULTS_ROOT/$name/train"
  local log_file="$LOG_ROOT/train_${name}.log"
  local ckpt_path="$outdir/ckpt.ckpt"

  if [[ $RESUME -eq 1 && -f "$ckpt_path" ]]; then
    log "Skipping $name (ckpt exists)."
    return 0
  fi

  mkdir -p "$outdir" "$outdir/tmp" "$outdir/wandb"
  log "Launching $name on GPU $gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TMPDIR="$outdir/tmp"
    export WANDB_DIR="$outdir/wandb"
    export WANDB_DISABLE_SERVICE=1
    export WANDB_USE=0
    export WANDB_DISABLED=1
    unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH \
      WANDB_CONFIG WANDB_RUN_ID
    python federatedscope/main.py \
      --cfg "$cfg" \
      outdir "$outdir" \
      expname "$name" \
      expname_tag "run_${PIPE_ID}" \
      federate.save_to "$ckpt_path" \
      device 0
  ) > "$log_file" 2>&1 &
  LAST_PID=$!
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

JOBS=(
  "proj_on|$CFG_ON"
  "proj_off|$CFG_OFF"
)

job_index=0
total_jobs=${#JOBS[@]}
gpu_count=${#GPUS[@]}

while [[ $job_index -lt $total_jobs ]]; do
  pids=()
  for ((i=0; i<gpu_count && job_index<total_jobs; i++)); do
    IFS='|' read -r name cfg <<< "${JOBS[$job_index]}"
    run_job "$name" "$cfg" "${GPUS[$i]}"
    if [[ -n "${LAST_PID:-}" ]]; then
      pids+=("$LAST_PID")
    fi
    job_index=$((job_index + 1))
  done
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}"
  fi
done

log "Projection sanity runs complete."
