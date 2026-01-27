#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_COLLECT=0
RESUME=0
PRINT_EVAL_COMMANDS=0
PRINT_EVAL_ON_TRAIN_COMPLETE=0
EVAL_SERVERS=4
EVAL_GPUS="0,1,2,3,4,5,6,7"
GPUS="0,1,2,3,4,5,6,7"
PIPE_ID="${INDIVIDUAL_ABLATION_ID:-}"
EVAL_ID="${INDIVIDUAL_ABLATION_EVAL_ID:-}"
CONFIG_DIR="${INDIVIDUAL_ABLATION_CONFIG_DIR:-final/configs/ablations}"
RESULTS_ROOT="${INDIVIDUAL_ABLATION_RESULTS_DIR:-final/results/ablations}"
LOG_ROOT="${INDIVIDUAL_ABLATION_LOGS_DIR:-final/logs/ablations}"
DATA_ROOT="${INDIVIDUAL_DATA_ROOT:-data/individual_federated}"
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-100}"
TRAIN_OPTS="${INDIVIDUAL_TRAIN_OPTS:-}"
RUN_TAGS_OVERRIDE=""
CONDA_ENV="${INDIVIDUAL_CONDA_ENV:-}"
HF_HOME_OVERRIDE=""
FULL_EVAL=0

usage() {
  cat <<'USAGE'
Usage: run_local_individual_ablation_pipeline.sh [options]

Options:
  --dry-run              Print commands without running them.
  --skip-train           Skip training jobs and go straight to eval (if enabled).
  --skip-eval            Skip evaluation jobs.
  --skip-collect         Skip metrics aggregation.
  --resume               Skip steps with existing outputs.
  --print-eval-commands  Print eval-only commands split across servers.
  --print-eval-on-train-complete
                         Print an eval-only command as each training job finishes.
  --eval-servers         Number of eval servers to split work across (default: 4).
  --eval-gpus            GPU list to use for eval commands (default: 0-7).
  --gpus                 Comma-separated GPU list (default: 0,1,2,3,4,5,6,7).
  --pipe-id              Override pipeline id (default: auto).
  --eval-id              Override eval job id (default: eval_<pipe-id>).
  --config-dir           Ablation YAML directory (default: final/configs/ablations).
  --results-root         Results root (default: final/results/ablations).
  --logs-root            Logs root (default: final/logs/ablations).
  --data-root            Dataset root (default: data/individual_federated).
  --eval-max-samples     Max samples per eval task (omit for full eval).
  --full-eval            Run full benchmark (ignore eval-max-samples).
  --run-tags             Comma-separated ablation tags to run (default: all).
  --train-opts           Extra federatedscope overrides (use :: as separator).
  --conda-env            Conda env to activate before running.
  --hf-home              Override HF_HOME for dataset/model cache.
  -h, --help             Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --skip-collect)
      SKIP_COLLECT=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --print-eval-commands)
      PRINT_EVAL_COMMANDS=1
      shift
      ;;
    --print-eval-on-train-complete)
      PRINT_EVAL_ON_TRAIN_COMPLETE=1
      shift
      ;;
    --eval-servers)
      EVAL_SERVERS="$2"
      shift 2
      ;;
    --eval-gpus)
      EVAL_GPUS="$2"
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
    --eval-id)
      EVAL_ID="$2"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --logs-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
      shift 2
      ;;
    --full-eval)
      FULL_EVAL=1
      shift
      ;;
    --run-tags)
      RUN_TAGS_OVERRIDE="$2"
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

if [[ $FULL_EVAL -eq 1 ]]; then
  EVAL_MAX_SAMPLES=""
fi
if [[ "$EVAL_MAX_SAMPLES" == "all" || "$EVAL_MAX_SAMPLES" == "full" ]]; then
  EVAL_MAX_SAMPLES=""
fi
if [[ -n "$EVAL_MAX_SAMPLES" && ! "$EVAL_MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "Invalid --eval-max-samples (must be integer or omitted for full eval)." >&2
  exit 1
fi
if [[ -n "$EVAL_SERVERS" && ! "$EVAL_SERVERS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --eval-servers (must be integer)." >&2
  exit 1
fi

for var_name in CONFIG_DIR RESULTS_ROOT LOG_ROOT DATA_ROOT; do
  value="${!var_name}"
  if [[ "$value" == FTL/* ]]; then
    value="${value#FTL/}"
  fi
  if [[ "$value" != /* ]]; then
    value="$ROOT_DIR/$value"
  fi
  printf -v "$var_name" '%s' "$value"
done

DATA_ROOT_SUFFIX="$DATA_ROOT"
if [[ "$DATA_ROOT_SUFFIX" == $ROOT_DIR/* ]]; then
  DATA_ROOT_SUFFIX="${DATA_ROOT_SUFFIX#${ROOT_DIR}/}"
fi
if [[ "$DATA_ROOT_SUFFIX" == data/* ]]; then
  DATA_ROOT_SUFFIX="${DATA_ROOT_SUFFIX#data/}"
fi

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="individual_ablations_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="eval_${PIPE_ID}"
fi

RESULTS_DIR="$RESULTS_ROOT/$PIPE_ID"
LOG_DIR="$LOG_ROOT/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
: > "$PIPE_LOG"
printf "phase\texp\tlabel\tgpu\tstatus\tstart\tend\tlog\n" > "$STATUS_FILE"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export PYTHONUNBUFFERED=1

if [[ -z "$CONDA_ENV" && -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
  CONDA_ENV="$CONDA_PREFIX"
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
  printf "[%s] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")"     "conda not found; continuing without activation." >&2
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
JOB_EXPS=()

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

build_eval_command() {
  local exp="$1"
  local -a cmd=(
    bash "$ROOT_DIR/scripts/run_local_individual_ablation_pipeline.sh"
    --skip-train --resume --skip-collect
    --pipe-id "$PIPE_ID"
    --eval-id "$EVAL_ID"
    --run-tags "$exp"
    --gpus "$EVAL_GPUS"
    --config-dir "$CONFIG_DIR"
    --results-root "$RESULTS_ROOT"
    --logs-root "$LOG_ROOT"
  )
  if [[ $FULL_EVAL -eq 1 ]]; then
    cmd+=(--full-eval)
  elif [[ -n "$EVAL_MAX_SAMPLES" ]]; then
    cmd+=(--eval-max-samples "$EVAL_MAX_SAMPLES")
  fi
  if [[ -n "$CONDA_ENV" ]]; then
    cmd+=(--conda-env "$CONDA_ENV")
  fi
  if [[ -n "$HF_HOME_OVERRIDE" ]]; then
    cmd+=(--hf-home "$HF_HOME_OVERRIDE")
  fi
  local out=""
  local arg
  for arg in "${cmd[@]}"; do
    out+=$(printf '%q ' "$arg")
  done
  printf '%s' "${out% }"
}

record_status() {
  local phase="$1"
  local exp="$2"
  local label="$3"
  local gpu="$4"
  local status="$5"
  local start_ts="$6"
  local end_ts="$7"
  local log_file="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"     "$phase" "$exp" "$label" "$gpu" "$status"     "$start_ts" "$end_ts" "$log_file" >> "$STATUS_FILE"
}

run_job() {
  local phase="$1"
  local exp="$2"
  local label="$3"
  local log_file="$4"
  shift 4
  local eval_cmd=""
  if [[ "$phase" == "train" && $PRINT_EVAL_ON_TRAIN_COMPLETE -eq 1 ]]; then
    eval_cmd="$(build_eval_command "$exp")"
  fi

  local start_ts
  start_ts="$(ts)"
  if [[ $DRY_RUN -eq 1 ]]; then
    local gpu
    gpu="$(next_dry_gpu)"
    log "DRY-RUN [$phase][$exp] $label gpu=$gpu cmd: $*"
    record_status "$phase" "$exp" "$label" "$gpu"       "dry-run" "$start_ts" "$start_ts" "$log_file"
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
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@" >> "$log_file" 2>&1
    if [[ -n "$eval_cmd" ]]; then
      local ready_ts
      ready_ts="$(ts)"
      printf "[%s] Eval-ready command for %s: %s\n" "$ready_ts" "$exp" "$eval_cmd" >> "$PIPE_LOG"
      printf "%s\n" "$eval_cmd" >> "$LOG_DIR/eval_commands_ready.txt"
    fi
  ) &
  local pid=$!
  JOB_PIDS+=("$pid")
  JOB_LABELS+=("$label")
  JOB_LOGS+=("$log_file")
  JOB_GPUS+=("$gpu")
  JOB_STARTS+=("$start_ts")
  JOB_PHASES+=("$phase")
  JOB_EXPS+=("$exp")
}

remove_job_at_index() {
  local idx="$1"
  unset 'JOB_PIDS[idx]'
  unset 'JOB_LABELS[idx]'
  unset 'JOB_LOGS[idx]'
  unset 'JOB_GPUS[idx]'
  unset 'JOB_STARTS[idx]'
  unset 'JOB_PHASES[idx]'
  unset 'JOB_EXPS[idx]'
  JOB_PIDS=("${JOB_PIDS[@]}")
  JOB_LABELS=("${JOB_LABELS[@]}")
  JOB_LOGS=("${JOB_LOGS[@]}")
  JOB_GPUS=("${JOB_GPUS[@]}")
  JOB_STARTS=("${JOB_STARTS[@]}")
  JOB_PHASES=("${JOB_PHASES[@]}")
  JOB_EXPS=("${JOB_EXPS[@]}")
}

wait_jobs() {
  local filter_phase="${1:-}"
  local failures=0
  if [[ ${#JOB_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  local completed_indices=()
  for idx in "${!JOB_PIDS[@]}"; do
    if [[ -n "$filter_phase" && "${JOB_PHASES[$idx]}" != "$filter_phase" ]]; then
      continue
    fi
    local pid="${JOB_PIDS[$idx]}"
    local label="${JOB_LABELS[$idx]}"
    local log_file="${JOB_LOGS[$idx]}"
    local gpu="${JOB_GPUS[$idx]}"
    local start_ts="${JOB_STARTS[$idx]}"
    local phase="${JOB_PHASES[$idx]}"
    local exp="${JOB_EXPS[$idx]}"
    local status=0
    if wait "$pid"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
    fi
    local end_ts
    end_ts="$(ts)"
    record_status "$phase" "$exp" "$label" "$gpu"       "$status" "$start_ts" "$end_ts" "$log_file"
    if [[ $status -ne 0 ]]; then
      log "Job failed: $label ($exp, gpu=$gpu, log=$log_file)"
    else
      log "Job finished: $label ($exp, gpu=$gpu)"
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

WAITED_JOB_LABEL=""
WAITED_JOB_STATUS=0
WAITED_JOB_EXP=""
WAITED_JOB_GPU=""
WAITED_JOB_LOG=""

wait_for_any_train_job() {
  while true; do
    for idx in "${!JOB_PIDS[@]}"; do
      if [[ "${JOB_PHASES[$idx]}" != "train" ]]; then
        continue
      fi
      local pid="${JOB_PIDS[$idx]}"
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      local label="${JOB_LABELS[$idx]}"
      local log_file="${JOB_LOGS[$idx]}"
      local gpu="${JOB_GPUS[$idx]}"
      local start_ts="${JOB_STARTS[$idx]}"
      local exp="${JOB_EXPS[$idx]}"
      local status=0
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      local end_ts
      end_ts="$(ts)"
      record_status "train" "$exp" "$label" "$gpu" \
        "$status" "$start_ts" "$end_ts" "$log_file"
      if [[ $status -ne 0 ]]; then
        log "Job failed: $label ($exp, gpu=$gpu, log=$log_file)"
      else
        log "Job finished: $label ($exp, gpu=$gpu)"
      fi
      WAITED_JOB_LABEL="$label"
      WAITED_JOB_STATUS="$status"
      WAITED_JOB_EXP="$exp"
      WAITED_JOB_GPU="$gpu"
      WAITED_JOB_LOG="$log_file"
      remove_job_at_index "$idx"
      return 0
    done
    sleep 1
  done
}

find_ckpt() {
  local result_dir="$1"
  local exp="$2"
  local base="$result_dir/$exp/train"
  if [[ -f "$base/final_ckpt.ckpt" ]]; then
    printf '%s' "$base/final_ckpt.ckpt"
  elif [[ -f "$base/ckpt.ckpt" ]]; then
    printf '%s' "$base/ckpt.ckpt"
  else
    printf ''
  fi
}

has_eval_result() {
  local res_dir="$1"
  if [[ ! -d "$res_dir" ]]; then
    return 1
  fi
  if find "$res_dir" -type f -name "accuracies_*.json" -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

if [[ $DRY_RUN -eq 0 ]]; then
  init_gpu_queue
fi

EXPS=(
  abl_full
  abl_fedavg
  abl_decomp_only
  abl_reweight_only
  abl_server_only
  abl_lora_A_only
  abl_lora_B_only
  abl_lora_AB
  abl_proj_rho_0
  abl_proj_rho_0p5
  abl_proj_mode_plus_shared
  abl_proj_rank_max_8
  abl_proj_rank_max_16
  abl_beta_g0_r0
  abl_beta_g0_r0p5
  abl_beta_g0_r1
  abl_beta_g0p5_r0
  abl_beta_g0p5_r0p5
  abl_beta_g0p5_r1
  abl_beta_g1_r0
  abl_beta_g1_r0p5
  abl_beta_g1_r1
)

if [[ -n "$RUN_TAGS_OVERRIDE" ]]; then
  IFS=',' read -r -a REQUESTED_RUNS <<< "${RUN_TAGS_OVERRIDE// /}"
  EXPS=()
  for tag in "${REQUESTED_RUNS[@]}"; do
    EXPS+=("$tag")
  done
fi

TASKS=(gsm8k hellaswag xsum hotpotqa mbpp)

emit_eval_commands() {
  local server_count="$1"
  local eval_gpus="$2"
  local cmd_file="$LOG_DIR/eval_commands.txt"
  local idx server
  local -a SERVER_TAGS=()
  local total="${#EXPS[@]}"

  if [[ "$server_count" -lt 1 ]]; then
    server_count=1
  fi

  for ((server=0; server<server_count; server++)); do
    SERVER_TAGS[$server]=""
  done
  for idx in "${!EXPS[@]}"; do
    server=$((idx % server_count))
    if [[ -n "${SERVER_TAGS[$server]}" ]]; then
      SERVER_TAGS[$server]="${SERVER_TAGS[$server]},${EXPS[$idx]}"
    else
      SERVER_TAGS[$server]="${EXPS[$idx]}"
    fi
  done

  : > "$cmd_file"
  log "Eval commands (servers=$server_count, total_exps=$total):"
  for ((server=0; server<server_count; server++)); do
    local run_tags="${SERVER_TAGS[$server]}"
    if [[ -z "$run_tags" ]]; then
      continue
    fi
    local cmd=(bash "$ROOT_DIR/scripts/run_local_individual_ablation_pipeline.sh"
      --skip-train --resume --skip-collect
      --pipe-id "$PIPE_ID"
      --eval-id "$EVAL_ID"
      --run-tags "$run_tags"
      --gpus "$eval_gpus"
      --config-dir "$CONFIG_DIR"
      --results-root "$RESULTS_ROOT"
      --logs-root "$LOG_ROOT"
    )
    if [[ $FULL_EVAL -eq 1 ]]; then
      cmd+=(--full-eval)
    elif [[ -n "$EVAL_MAX_SAMPLES" ]]; then
      cmd+=(--eval-max-samples "$EVAL_MAX_SAMPLES")
    fi
    if [[ -n "$CONDA_ENV" ]]; then
      cmd+=(--conda-env "$CONDA_ENV")
    fi
    if [[ -n "$HF_HOME_OVERRIDE" ]]; then
      cmd+=(--hf-home "$HF_HOME_OVERRIDE")
    fi
    log "Eval server $((server + 1)): ${cmd[*]}"
    printf "%s\n" "${cmd[*]}" >> "$cmd_file"
  done
  log "Saved eval commands to $cmd_file"
}

emit_eval_command_for_exp() {
  local exp="$1"
  local cmd_file="$LOG_DIR/eval_commands_ready.txt"
  local cmd=(bash "$ROOT_DIR/scripts/run_local_individual_ablation_pipeline.sh"
    --skip-train --resume --skip-collect
    --pipe-id "$PIPE_ID"
    --eval-id "$EVAL_ID"
    --run-tags "$exp"
    --gpus "$EVAL_GPUS"
    --config-dir "$CONFIG_DIR"
    --results-root "$RESULTS_ROOT"
    --logs-root "$LOG_ROOT"
  )
  if [[ $FULL_EVAL -eq 1 ]]; then
    cmd+=(--full-eval)
  elif [[ -n "$EVAL_MAX_SAMPLES" ]]; then
    cmd+=(--eval-max-samples "$EVAL_MAX_SAMPLES")
  fi
  if [[ -n "$CONDA_ENV" ]]; then
    cmd+=(--conda-env "$CONDA_ENV")
  fi
  if [[ -n "$HF_HOME_OVERRIDE" ]]; then
    cmd+=(--hf-home "$HF_HOME_OVERRIDE")
  fi
  log "Eval-ready command for $exp: ${cmd[*]}"
  printf "%s\n" "${cmd[*]}" >> "$cmd_file"
}

BASE_TRAIN_OPTS="data.tulu3_federated.root ${DATA_ROOT_SUFFIX}"
if [[ -n "$TRAIN_OPTS" ]]; then
  FULL_TRAIN_OPTS="${BASE_TRAIN_OPTS}::${TRAIN_OPTS}"
else
  FULL_TRAIN_OPTS="$BASE_TRAIN_OPTS"
fi
TRAIN_OPT_ARGS=()
read -r -a TRAIN_OPT_ARGS <<< "${FULL_TRAIN_OPTS//::/ }"

TOTAL_FAILURES=0
train_jobs=0

log "Initialized ablation pipeline"
log "  pipe_id=$PIPE_ID eval_id=$EVAL_ID"
log "  config_dir=$CONFIG_DIR"
log "  results_dir=$RESULTS_DIR"
log "  log_dir=$LOG_DIR"
log "  data_root=$DATA_ROOT eval_max_samples=${EVAL_MAX_SAMPLES:-full}"
log "---"

if [[ $SKIP_TRAIN -eq 0 ]]; then
  log "Launching ablation training jobs..."
  for exp in "${EXPS[@]}"; do
    CFG="$CONFIG_DIR/${exp}.yaml"
    if [[ ! -f "$CFG" ]]; then
      log "Missing config for $exp: $CFG"
      TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
      continue
    fi
    OUTDIR="$RESULTS_DIR/$exp/train"
    CKPT_PATH="$OUTDIR/ckpt.ckpt"
    mkdir -p "$OUTDIR"
    TRAIN_LOG="$LOG_DIR/train_${exp}.log"
    EXISTING_CKPT="$(find_ckpt "$RESULTS_DIR" "$exp")"
    if [[ $RESUME -eq 1 && -n "$EXISTING_CKPT" ]]; then
      log "Skipping training for $exp (checkpoint exists)."
      ts_now="$(ts)"
      record_status "train" "$exp" "train-$exp" "-" "skipped" "$ts_now" "$ts_now" "$TRAIN_LOG"
      if [[ $PRINT_EVAL_ON_TRAIN_COMPLETE -eq 1 ]]; then
        emit_eval_command_for_exp "$exp"
      fi
      continue
    fi
    run_job "train" "$exp" "train-$exp" "$TRAIN_LOG" \
      bash -c "export TMPDIR='$OUTDIR/tmp' WANDB_DIR='$OUTDIR/wandb' \
        WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
        mkdir -p '$OUTDIR/tmp' '$OUTDIR/wandb'; \
        unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
        python federatedscope/main.py \
          --cfg '$CFG' \
          outdir '$OUTDIR' expname '$exp' expname_tag 'run_${PIPE_ID}' \
          federate.save_to '$CKPT_PATH' \
          ${TRAIN_OPT_ARGS[*]}"
    if [[ $DRY_RUN -eq 0 ]]; then
      train_jobs=$((train_jobs + 1))
    fi
  done

  if [[ $DRY_RUN -eq 0 ]]; then
    while [[ $train_jobs -gt 0 ]]; do
      wait_for_any_train_job
      train_jobs=$((train_jobs - 1))
      if [[ $WAITED_JOB_STATUS -ne 0 ]]; then
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
      elif [[ $PRINT_EVAL_ON_TRAIN_COMPLETE -eq 1 ]]; then
        emit_eval_command_for_exp "$WAITED_JOB_EXP"
      fi
    done
  fi
else
  log "Skipping training stage."
fi

if [[ $PRINT_EVAL_COMMANDS -eq 1 ]]; then
  emit_eval_commands "$EVAL_SERVERS" "$EVAL_GPUS"
fi

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Launching ablation eval jobs (after training)..."
  for exp in "${EXPS[@]}"; do
    CKPT_PATH="$(find_ckpt "$RESULTS_DIR" "$exp")"
    if [[ -z "$CKPT_PATH" ]]; then
      log "No checkpoint found for $exp; skipping eval."
      continue
    fi
    for task in "${TASKS[@]}"; do
      RES_DIR="$RESULTS_DIR/global/$exp/$task/$EVAL_ID"
      EVAL_LOG="$LOG_DIR/eval_${exp}_${task}.log"
      EVAL_YAML="$RES_DIR/eval_${exp}_${task}.yaml"
      mkdir -p "$RES_DIR"
      if [[ $RESUME -eq 1 ]] && has_eval_result "$RES_DIR"; then
        log "Skipping eval for $exp/$task (results exist)."
        ts_now="$(ts)"
        record_status "eval" "$exp" "eval-$exp-$task" "-" "skipped" "$ts_now" "$ts_now" "$EVAL_LOG"
        continue
      fi

      max_new_tokens=64
      case "$task" in
        gsm8k)
          max_new_tokens=256
          ;;
        hellaswag)
          max_new_tokens=8
          ;;
        xsum)
          max_new_tokens=128
          ;;
        hotpotqa)
          max_new_tokens=32
          ;;
        mbpp)
          max_new_tokens=256
          ;;
      esac

      cat > "$EVAL_YAML" <<'EVAL_CFG'
use_gpu: True
device: 0
outdir: "__RES_DIR__/exp"
federate:
  save_to: "__CKPT__"
model:
  type: "meta-llama/Llama-2-7b-hf@huggingface_llm"
llm:
  tok_len: 2048
  adapter:
    use: True
    args:
      - {adapter_package: "peft", adapter_method: "lora",
         r: 8, lora_alpha: 32, lora_dropout: 0.05,
         target_modules: ["q_proj","k_proj","v_proj","o_proj"],
         modules_to_save: ["embed_tokens","lm_head"]}
train:
  is_enable_half: False
  precision: bf16
  compile: False
eval:
  max_samples: __MAX_SAMPLES__
  max_new_tokens: __MAX_NEW_TOKENS__
  num_completions: 1
  timeout: 5
  max_samples_per_subject: 2
  superglue_tasks: ["boolq", "rte", "cb", "copa", "wic"]
EVAL_CFG
      sed -i \
        -e "s#__RES_DIR__#$RES_DIR#g" \
        -e "s#__CKPT__#$CKPT_PATH#g" \
        -e "s#__MAX_NEW_TOKENS__#$max_new_tokens#g" \
        "$EVAL_YAML"
      if [[ -n "$EVAL_MAX_SAMPLES" ]]; then
        sed -i "s#__MAX_SAMPLES__#$EVAL_MAX_SAMPLES#g" "$EVAL_YAML"
      else
        sed -i "/max_samples:/d" "$EVAL_YAML"
      fi

      eval_cmd=()
      case "$task" in
        gsm8k)
          eval_cmd=(python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$EVAL_YAML")
          ;;
        hellaswag)
          eval_cmd=(python federatedscope/llm/eval/eval_for_hellaswag/eval.py --cfg "$EVAL_YAML")
          ;;
        xsum)
          eval_cmd=(python federatedscope/llm/eval/eval_for_xsum/eval.py --cfg "$EVAL_YAML")
          ;;
        hotpotqa)
          eval_cmd=(python federatedscope/llm/eval/eval_for_hotpotqa/eval.py --cfg "$EVAL_YAML")
          ;;
        mbpp)
          eval_cmd=(python federatedscope/llm/eval/eval_for_mbpp/eval.py --cfg "$EVAL_YAML")
          ;;
        *)
          log "Unknown task: $task"
          continue
          ;;
      esac

      run_job "eval" "$exp" "eval-$exp-$task" "$EVAL_LOG" \
        bash -c "TMP_BASE='${RESULTS_DIR}/tmp'; \
          mkdir -p '\$TMP_BASE' '$RES_DIR/wandb'; \
          export TMPDIR='\$TMP_BASE/${EVAL_ID}_${exp}_${task}' WANDB_DIR='$RES_DIR/wandb' WANDB_DISABLE_SERVICE=1; \
          unset WANDB_SERVICE; \
          ${eval_cmd[*]}"
    done
  done

  log "Waiting for evaluations..."
  if ! wait_jobs "eval"; then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
  fi

  if [[ $SKIP_COLLECT -eq 0 ]]; then
    log "Collecting metrics..."
    for exp in "${EXPS[@]}"; do
      COLLECT_LOG="$LOG_DIR/collect_${exp}.log"
      if [[ $DRY_RUN -eq 1 ]]; then
        log "DRY-RUN [collect-$exp] python scripts/collect_tulu_eval_results.py --results-root $RESULTS_DIR --exp $exp --eval-job-id $EVAL_ID"
        continue
      fi
      : > "$COLLECT_LOG"
      COLLECT_CMD=(python scripts/collect_tulu_eval_results.py
        --results-root "$RESULTS_DIR"
        --exp "$exp"
        --eval-job-id "$EVAL_ID"
      )
      log "Running metrics collection for $exp (log: $COLLECT_LOG)"
      (export TMPDIR="$RESULTS_DIR/tmp" WANDB_DIR="$RESULTS_DIR/wandb" WANDB_DISABLE_SERVICE=1; \
        mkdir -p "$TMPDIR" "$WANDB_DIR"; \
        unset WANDB_SERVICE; \
        "${COLLECT_CMD[@]}") >> "$COLLECT_LOG" 2>&1
    done
  fi
else
  log "Skipping evaluations."
fi

log "Ablation pipeline complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs executed."
fi
if [[ $TOTAL_FAILURES -ne 0 ]]; then
  log "Pipeline finished with failures. See $STATUS_FILE for details."
  exit 1
fi
