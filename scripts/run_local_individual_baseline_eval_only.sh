#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
SKIP_COLLECT=0
GPUS="0,1,2,3,4,5,6,7"
PIPE_ID="${INDIVIDUAL_BASELINE_EVAL_PIPE_ID:-}"
EVAL_ID="${INDIVIDUAL_BASELINE_EVAL_ID:-}"
BASELINE_DIR="${INDIVIDUAL_BASELINE_DIR:-individual_results/individual_local_20260115_100457_21987}"
RESULTS_ROOT="${INDIVIDUAL_FINAL_RESULTS_ROOT:-final/results/single_client}"
LOG_ROOT="${INDIVIDUAL_FINAL_LOG_ROOT:-final/logs/single_client}"
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-200}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_individual_baseline_eval_only.sh [options]

Options:
  --dry-run             Print commands without running them.
  --resume              Skip evals with existing outputs.
  --skip-collect         Skip metrics aggregation.
  --gpus                 Comma-separated GPU list (default: 0-7).
  --pipe-id              Override pipeline id (default: auto).
  --eval-id              Override evaluation id base (default: eval_<pipe-id>).
  --baseline-dir          Baseline results dir for checkpoints.
  --results-root          Final results root (default: final/results/single_client).
  --log-root              Final logs root (default: final/logs/single_client).
  --eval-max-samples       Max samples per eval task (default: 200).
  --conda-env             Conda env to activate before running.
  --hf-home               Override HF_HOME for dataset/model cache.
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
    --skip-collect)
      SKIP_COLLECT=1
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
    --eval-id)
      EVAL_ID="$2"
      shift 2
      ;;
    --baseline-dir)
      BASELINE_DIR="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
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

normalize_path() {
  local path="$1"
  if [[ "$path" == FTL/* ]]; then
    path="${path#FTL/}"
  fi
  if [[ "$path" != /* ]]; then
    path="$ROOT_DIR/$path"
  fi
  printf '%s' "$path"
}

BASELINE_DIR="$(normalize_path "$BASELINE_DIR")"
RESULTS_ROOT="$(normalize_path "$RESULTS_ROOT")"
LOG_ROOT="$(normalize_path "$LOG_ROOT")"

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="individual_baseline_eval_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="eval_${PIPE_ID}"
fi

LOG_DIR="$LOG_ROOT/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"
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
    log "DRY-RUN [$phase] $label gpu=$gpu cmd: $*"
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
  JOB_PIDS=("${JOB_PIDS[@]}")
  JOB_LABELS=("${JOB_LABELS[@]}")
  JOB_LOGS=("${JOB_LOGS[@]}")
  JOB_GPUS=("${JOB_GPUS[@]}")
  JOB_STARTS=("${JOB_STARTS[@]}")
  JOB_PHASES=("${JOB_PHASES[@]}")
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

find_ckpt_in_dir() {
  local train_dir="$1"
  if [[ -f "$train_dir/final_ckpt.ckpt" ]]; then
    printf '%s' "$train_dir/final_ckpt.ckpt"
  elif [[ -f "$train_dir/ckpt.ckpt" ]]; then
    printf '%s' "$train_dir/ckpt.ckpt"
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

write_eval_yaml() {
  local res_dir="$1"
  local ckpt_path="$2"
  local eval_yaml="$3"
  local max_samples="$4"
  local max_new_tokens="$5"
  local num_completions="$6"
  local timeout_sec="$7"
  local mmlu_samples="$8"
  cat > "$eval_yaml" <<'EVAL_CFG'
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
  num_completions: __NUM_COMPLETIONS__
  timeout: __TIMEOUT__
  max_samples_per_subject: __MMLU_SAMPLES__
  superglue_tasks: ["boolq", "rte", "cb", "copa", "wic"]
EVAL_CFG
  sed -i \
    -e "s#__RES_DIR__#$res_dir#g" \
    -e "s#__CKPT__#$ckpt_path#g" \
    -e "s#__MAX_SAMPLES__#$max_samples#g" \
    -e "s#__MAX_NEW_TOKENS__#$max_new_tokens#g" \
    -e "s#__NUM_COMPLETIONS__#$num_completions#g" \
    -e "s#__TIMEOUT__#$timeout_sec#g" \
    -e "s#__MMLU_SAMPLES__#$mmlu_samples#g" \
    "$eval_yaml"
}

max_new_tokens_for_task() {
  case "$1" in
    gsm8k)
      printf '256'
      ;;
    hellaswag)
      printf '8'
      ;;
    xsum)
      printf '128'
      ;;
    hotpotqa)
      printf '32'
      ;;
    mbpp)
      printf '256'
      ;;
    *)
      printf '64'
      ;;
  esac
}

schedule_eval() {
  local exp="$1"
  local task="$2"
  local ckpt_path="$3"

  if [[ ! -f "$ckpt_path" ]]; then
    log "No checkpoint for $exp/$task: $ckpt_path"
    return 0
  fi

  local res_dir="$RESULTS_ROOT/global/$exp/$task/$EVAL_ID"
  local eval_log="$LOG_DIR/eval_${exp}_${task}.log"

  if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
    log "Skipping eval for $exp/$task (results exist)."
    local ts_now
    ts_now="$(ts)"
    record_status "eval" "eval-$exp-$task" "-" "skipped" "$ts_now" "$ts_now" "$eval_log"
    return 0
  fi

  mkdir -p "$res_dir"
  local eval_yaml="$res_dir/eval_${exp}_${task}.yaml"
  local max_samples="$EVAL_MAX_SAMPLES"
  if [[ -z "$max_samples" ]]; then
    max_samples=200
  fi
  local max_new_tokens
  max_new_tokens="$(max_new_tokens_for_task "$task")"
  local num_completions=1
  local timeout_sec=5
  local mmlu_samples=2

  write_eval_yaml "$res_dir" "$ckpt_path" "$eval_yaml" \
    "$max_samples" "$max_new_tokens" "$num_completions" "$timeout_sec" "$mmlu_samples"

  local eval_cmd=()
  case "$task" in
    gsm8k)
      eval_cmd=(python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$eval_yaml")
      ;;
    hellaswag)
      eval_cmd=(python federatedscope/llm/eval/eval_for_hellaswag/eval.py --cfg "$eval_yaml")
      ;;
    xsum)
      eval_cmd=(python federatedscope/llm/eval/eval_for_xsum/eval.py --cfg "$eval_yaml")
      ;;
    hotpotqa)
      eval_cmd=(python federatedscope/llm/eval/eval_for_hotpotqa/eval.py --cfg "$eval_yaml")
      ;;
    mbpp)
      eval_cmd=(python federatedscope/llm/eval/eval_for_mbpp/eval.py --cfg "$eval_yaml")
      ;;
    *)
      log "Unknown task: $task"
      return 0
      ;;
  esac

  run_job "eval" "eval-$exp-$task" "$eval_log" \
    bash -c "TMP_BASE='$RESULTS_ROOT/tmp'; \
      mkdir -p \"\$TMP_BASE\" '$res_dir/wandb'; \
      export TMPDIR=\"\$TMP_BASE/${EVAL_ID}_${exp}_${task}\" \
        WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1; \
      unset WANDB_SERVICE; \
      ${eval_cmd[*]}"
}

if [[ $DRY_RUN -eq 0 ]]; then
  init_gpu_queue
fi

EXPS=(centralized fedavg bank_perclient)
TASKS=(gsm8k hellaswag xsum hotpotqa mbpp)
declare -A EXP_CKPT=()

for exp in "${EXPS[@]}"; do
  train_dir="$BASELINE_DIR/$exp/train"
  ckpt_path="$(find_ckpt_in_dir "$train_dir")"
  if [[ -z "$ckpt_path" ]]; then
    log "No checkpoint found for $exp in $train_dir; skipping evals."
    continue
  fi
  EXP_CKPT["$exp"]="$ckpt_path"
done

FAILURES=0
log "Launching single-client baseline evals (higher samples)..."
for exp in "${EXPS[@]}"; do
  ckpt_path="${EXP_CKPT[$exp]:-}"
  if [[ -z "$ckpt_path" ]]; then
    continue
  fi
  for task in "${TASKS[@]}"; do
    schedule_eval "$exp" "$task" "$ckpt_path"
  done
done

log "Waiting for baseline evals..."
if ! wait_jobs "eval"; then
  FAILURES=$((FAILURES + 1))
fi

if [[ $SKIP_COLLECT -eq 0 ]]; then
  log "Collecting baseline metrics..."
  for exp in "${EXPS[@]}"; do
    if [[ -z "${EXP_CKPT[$exp]:-}" ]]; then
      continue
    fi
    COLLECT_LOG="$LOG_DIR/collect_${exp}.log"
    if [[ $DRY_RUN -eq 1 ]]; then
      log "DRY-RUN [collect-$exp] python scripts/collect_tulu_eval_results.py --results-root $RESULTS_ROOT --exp $exp --eval-job-id $EVAL_ID"
      continue
    fi
    : > "$COLLECT_LOG"
    COLLECT_CMD=(python scripts/collect_tulu_eval_results.py
      --results-root "$RESULTS_ROOT"
      --exp "$exp"
      --eval-job-id "$EVAL_ID"
    )
    if [[ "$exp" == "bank_perclient" ]]; then
      COLLECT_CMD+=(--fedavg-eval-job-id "$EVAL_ID")
    fi
    log "Running metrics collection for $exp (log: $COLLECT_LOG)"
    (export TMPDIR="$RESULTS_ROOT/tmp" WANDB_DIR="$RESULTS_ROOT/wandb" WANDB_DISABLE_SERVICE=1; \
      mkdir -p "$TMPDIR" "$WANDB_DIR"; \
      unset WANDB_SERVICE; \
      "${COLLECT_CMD[@]}") >> "$COLLECT_LOG" 2>&1
  done
fi

log "Single-client baseline eval-only pipeline complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs executed."
fi
if [[ $FAILURES -ne 0 ]]; then
  log "Pipeline finished with failures. See $STATUS_FILE for details."
  exit 1
fi
