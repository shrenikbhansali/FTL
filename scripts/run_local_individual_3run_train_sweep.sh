#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
GPUS="0,1,2,3"
PIPE_ID="${INDIVIDUAL_PIPE_ID:-}"
DATA_ROOT="${INDIVIDUAL_DATA_ROOT:-data/individual_federated}"
TRAIN_OPTS="${INDIVIDUAL_TRAIN_OPTS:-}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""
RUN_TAGS_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_individual_3run_train_sweep.sh [options]

Options:
  --dry-run        Print commands without running them.
  --resume         Skip training jobs with existing checkpoints.
  --gpus           Comma-separated GPU list (default: 0,1,2,3).
  --pipe-id        Override pipeline id (default: auto).
  --run-tags       Comma-separated run tags to execute (default: all).
  --data-root      Dataset root (default: data/individual_federated).
  --train-opts     Extra federatedscope overrides (use :: as separator).
  --conda-env      Conda env to activate before running.
  --hf-home        Override HF_HOME for dataset/model cache.
  -h, --help       Show this message.
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
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --pipe-id)
      PIPE_ID="$2"
      shift 2
      ;;
    --run-tags)
      RUN_TAGS_OVERRIDE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --train-opts)
      TRAIN_OPTS="$2"
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

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="3run_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

BASE_DIR="$ROOT_DIR/3multiconfig"
LOG_DIR="$BASE_DIR/logs/$PIPE_ID"
RESULTS_DIR="$BASE_DIR/results/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
: > "$PIPE_LOG"
printf "phase\tlabel\tgpu\tstatus\tstart\tend\tlog\n" > "$STATUS_FILE"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export PYTHONUNBUFFERED=1

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
  printf "[%s] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" \
    "conda not found; continuing without activation." >&2
}

ensure_conda_libs
maybe_activate_conda "$CONDA_ENV"
ensure_conda_libs

GPUS="${GPUS// /}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

GPU_QUEUE_FD=""
DRY_RUN_COUNT=0
JOB_PIDS=()
JOB_LABELS=()
JOB_LOGS=()
JOB_GPUS=()
JOB_STARTS=()
JOB_PHASES=()

init_gpu_queue() {
  local fifo
  fifo="$(mktemp -u)"
  mkfifo "$fifo"
  exec {GPU_QUEUE_FD}<>"$fifo"
  rm -f "$fifo"
  for gpu in "${GPU_LIST[@]}"; do
    printf '%s\n' "$gpu" >&"$GPU_QUEUE_FD"
  done
}

next_dry_gpu() {
  local gpu="${GPU_LIST[$((DRY_RUN_COUNT % ${#GPU_LIST[@]}))]}"
  DRY_RUN_COUNT=$((DRY_RUN_COUNT + 1))
  printf '%s' "$gpu"
}

acquire_gpu() {
  local gpu
  read -r gpu <&"$GPU_QUEUE_FD"
  printf '%s' "$gpu"
}

release_gpu() {
  printf '%s\n' "$1" >&"$GPU_QUEUE_FD"
}

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  printf "[%s] %s\n" "$(ts)" "$*" | tee -a "$PIPE_LOG" >&2
}

record_status() {
  local phase="$1"
  local label="$2"
  local gpu="$3"
  local status="$4"
  local start_ts="$5"
  local end_ts="$6"
  local log_file="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$label" "$gpu" "$status" "$start_ts" "$end_ts" "$log_file" \
    >> "$STATUS_FILE"
}

run_job() {
  local phase="$1"
  local label="$2"
  local log_file="$3"
  shift 3
  local start_ts
  start_ts="$(ts)"

  if [[ $DRY_RUN -eq 1 ]]; then
    local gpu
    gpu="$(next_dry_gpu)"
    log "DRY-RUN [$label] gpu=$gpu cmd: $*"
    record_status "$phase" "$label" "$gpu" "dry-run" "$start_ts" "$start_ts" "$log_file"
    return 0
  fi

  local gpu
  gpu="$(acquire_gpu)"
  : > "$log_file"
  (
    set -euo pipefail
    cleanup() {
      local status=$?
      set +e
      release_gpu "$gpu"
      printf "[%s] END status=%s\n" "$(ts)" "$status" >> "$log_file"
      exit "$status"
    }
    trap cleanup EXIT
    printf "[%s] START gpu=%s\n" "$(ts)" "$gpu" >> "$log_file"
    printf "[%s] CMD: %s\n" "$(ts)" "$*" >> "$log_file"
    ensure_conda_libs
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@" >> "$log_file" 2>&1
  ) &

  local pid=$!
  JOB_PIDS+=("$pid")
  JOB_LABELS+=("$label")
  JOB_LOGS+=("$log_file")
  JOB_GPUS+=("$gpu")
  JOB_STARTS+=("$start_ts")
  JOB_PHASES+=("$phase")
}

remove_job_at_index() {
  local idx="$1"
  unset 'JOB_PIDS[idx]'
  unset 'JOB_LABELS[idx]'
  unset 'JOB_LOGS[idx]'
  unset 'JOB_GPUS[idx]'
  unset 'JOB_STARTS[idx]'
  unset 'JOB_PHASES[idx]'
}

wait_jobs() {
  local failures=0
  if [[ ${#JOB_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  local completed_indices=()
  for idx in "${!JOB_PIDS[@]}"; do
    local pid="${JOB_PIDS[$idx]}"
    local label="${JOB_LABELS[$idx]}"
    local log_file="${JOB_LOGS[$idx]}"
    local gpu="${JOB_GPUS[$idx]}"
    local start_ts="${JOB_STARTS[$idx]}"
    local phase="${JOB_PHASES[$idx]}"
    local status=0
    if wait "$pid"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
    fi
    local end_ts
    end_ts="$(ts)"
    record_status "$phase" "$label" "$gpu" "$status" "$start_ts" "$end_ts" "$log_file"
    if [[ $status -ne 0 ]]; then
      log "Job failed: $label (gpu=$gpu, log=$log_file)"
    else
      log "Job finished: $label (gpu=$gpu)"
    fi
    completed_indices+=("$idx")
  done
  for ((i=${#completed_indices[@]}-1; i>=0; i--)); do
    remove_job_at_index "${completed_indices[$i]}"
  done
  if [[ $failures -ne 0 ]]; then
    return 1
  fi
  return 0
}

find_ckpt() {
  local run_tag="$1"
  local exp="$2"
  local base="$RESULTS_DIR/$run_tag/$exp/train"
  if [[ -f "$base/final_ckpt.ckpt" ]]; then
    printf '%s' "$base/final_ckpt.ckpt"
  elif [[ -f "$base/ckpt.ckpt" ]]; then
    printf '%s' "$base/ckpt.ckpt"
  else
    printf ''
  fi
}

if [[ $DRY_RUN -eq 0 ]]; then
  init_gpu_queue
fi

EXPS=(fedavg bank_perclient centralized)
RUN_TAGS=(participation_vlow participation_mid non_iid_low)
declare -A CFG_PREFIXES=(
  [participation_vlow]="individual_federated_sharded_participation_vlow"
  [participation_mid]="individual_federated_sharded_participation_mid"
  [non_iid_low]="individual_federated_sharded_non_iid_low"
)

if [[ -n "$RUN_TAGS_OVERRIDE" ]]; then
  IFS=',' read -r -a RUN_TAGS <<< "${RUN_TAGS_OVERRIDE// /}"
fi

BASE_TRAIN_OPTS="data.tulu3_federated.root ${DATA_ROOT#data/}"
if [[ -n "$TRAIN_OPTS" ]]; then
  FULL_TRAIN_OPTS="${BASE_TRAIN_OPTS}::${TRAIN_OPTS}"
else
  FULL_TRAIN_OPTS="$BASE_TRAIN_OPTS"
fi
TRAIN_OPT_ARGS=()
read -r -a TRAIN_OPT_ARGS <<< "${FULL_TRAIN_OPTS//::/ }"

log "Launching 3run training jobs..."

for run_tag in "${RUN_TAGS[@]}"; do
  cfg_prefix="${CFG_PREFIXES[$run_tag]}"
  if [[ -z "$cfg_prefix" ]]; then
    log "Unknown run tag: $run_tag"
    continue
  fi
  for exp in "${EXPS[@]}"; do
    train_dir="$RESULTS_DIR/$run_tag/$exp/train"
    train_log="$LOG_DIR/$run_tag/train_${exp}.log"
    mkdir -p "$train_dir" "$(dirname "$train_log")"

    if [[ $RESUME -eq 1 ]]; then
      ckpt_path="$(find_ckpt "$run_tag" "$exp")"
      if [[ -n "$ckpt_path" ]]; then
        ts_now="$(ts)"
        record_status "train" "train-$run_tag-$exp" "-" "skipped" "$ts_now" "$ts_now" "$train_log"
        log "Skipping train-$run_tag-$exp; checkpoint exists."
        continue
      fi
    fi

    run_job "train" "train-$run_tag-$exp" "$train_log" \
      bash -c "export TMPDIR='$train_dir/tmp' WANDB_DIR='$train_dir/wandb' \
        WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
        mkdir -p '$train_dir/tmp' '$train_dir/wandb'; \
        unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
        python federatedscope/main.py \
          --cfg 'yamls/${cfg_prefix}_${exp}.yaml' \
          outdir '$train_dir' \
          expname '$exp' \
          expname_tag 'run_${PIPE_ID}_${run_tag}' \
          federate.save_to '$train_dir/ckpt.ckpt' \
          ${TRAIN_OPT_ARGS[*]}"
  done
done

if ! wait_jobs; then
  log "Training sweep finished with failures. See $STATUS_FILE for details."
  exit 1
fi

log "3run training sweep complete."
