#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
GPUS="0,1,2,3,4,5,6,7"
SERVER_COUNT=1
SERVER_IDX=0
RUN_ID_OVERRIDE=""
RUN_TAGS_OVERRIDE=""
INCLUDE_GSM8K=0
SKIP_TOOLBENCH_IF_UNSET=1
SKIP_TASKS_OVERRIDE=""
FORCE_RERUN=0
STALE_HOURS=0
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-100}"
RESULTS_ROOT=""
LOGS_ROOT=""
CONDA_ENV=""
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_5multiconfig_eval_fanout.sh [options]

Options:
  --dry-run          Print planned jobs without running them.
  --resume           Skip evals with existing results (default: true).
  --gpus             Comma-separated GPU list (default: 0-7).
  --server-count     Total number of servers (default: 1).
  --server-idx       This server index [0..server-count-1] (default: 0).
  --run-id           Restrict to a single 5multiconfig run id.
  --run-tags         Comma-separated run tags (subdirs under run id).
  --include-gsm8k    Include GSM8K evals (default: off).
  --skip-tasks       Comma-separated tasks to skip (e.g. piqa,apps).
  --eval-max-samples Max samples per eval task (default: 100).
  --results-root     Override results root (default: FTL/5multiconfig/results).
  --logs-root        Override logs root (default: FTL/5multiconfig/logs).
  --force            Re-run even if eval log looks running/failed.
  --stale-hours      Consider running logs older than N hours as stale.
  --conda-env        Conda env to activate before running.
  --hf-home          Override HF_HOME for dataset/model cache.
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
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --server-count)
      SERVER_COUNT="$2"
      shift 2
      ;;
    --server-idx)
      SERVER_IDX="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID_OVERRIDE="$2"
      shift 2
      ;;
    --run-tags)
      RUN_TAGS_OVERRIDE="$2"
      shift 2
      ;;
    --include-gsm8k)
      INCLUDE_GSM8K=1
      shift
      ;;
    --skip-tasks)
      SKIP_TASKS_OVERRIDE="$2"
      shift 2
      ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
      shift 2
      ;;
    --force)
      FORCE_RERUN=1
      shift
      ;;
    --stale-hours)
      STALE_HOURS="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --logs-root)
      LOGS_ROOT="$2"
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

if [[ -z "$RESULTS_ROOT" ]]; then
  RESULTS_ROOT="$ROOT_DIR/5multiconfig/results"
fi
if [[ -z "$LOGS_ROOT" ]]; then
  LOGS_ROOT="$ROOT_DIR/5multiconfig/logs"
fi

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
  printf "[%s] %s\n" "$(ts)" "$*" >&2
}

log_grep() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings "$pattern" "$file"
  else
    grep -qF "$pattern" "$file"
  fi
}

run_job() {
  local label="$1"
  local log_file="$2"
  shift 2

  local start_ts
  start_ts="$(ts)"
  if [[ $DRY_RUN -eq 1 ]]; then
    local gpu
    gpu="$(next_dry_gpu)"
    log "DRY-RUN [$label] gpu=$gpu cmd: $*"
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
  JOB_PHASES+=("eval")
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
    local status=0
    if wait "$pid"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
    fi
    if [[ $status -ne 0 ]]; then
      log "Job failed: $label (gpu=$gpu, log=$log_file)"
    else
      log "Job finished: $label (gpu=$gpu)"
    fi
    completed_indices+=("$idx")
  done
  for ((i=${#completed_indices[@]}-1; i>=0; i--)); do
    unset 'JOB_PIDS[completed_indices[i]]'
    unset 'JOB_LABELS[completed_indices[i]]'
    unset 'JOB_LOGS[completed_indices[i]]'
    unset 'JOB_GPUS[completed_indices[i]]'
    unset 'JOB_STARTS[completed_indices[i]]'
    unset 'JOB_PHASES[completed_indices[i]]'
  done
  JOB_PIDS=("${JOB_PIDS[@]}")
  JOB_LABELS=("${JOB_LABELS[@]}")
  JOB_LOGS=("${JOB_LOGS[@]}")
  JOB_GPUS=("${JOB_GPUS[@]}")
  JOB_STARTS=("${JOB_STARTS[@]}")
  JOB_PHASES=("${JOB_PHASES[@]}")
  if [[ $failures -ne 0 ]]; then
    return 1
  fi
  return 0
}

find_ckpt() {
  local run_dir="$1"
  local exp="$2"
  local base="$run_dir/$exp/train"
  if [[ -f "$base/final_ckpt.ckpt" ]]; then
    printf '%s' "$base/final_ckpt.ckpt"
  elif [[ -f "$base/ckpt.ckpt" ]]; then
    printf '%s' "$base/ckpt.ckpt"
  else
    printf ''
  fi
}

find_accuracy() {
  local run_dir="$1"
  local exp="$2"
  local task="$3"
  local base="$run_dir/global/$exp/$task"
  if [[ ! -d "$base" ]]; then
    return 1
  fi
  if find "$base" -type f -name "accuracies_*__${task}.json" -print -quit | grep -q .; then
    return 0
  fi
  return 1
}

eval_log_state() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    printf 'missing'
    return 0
  fi
  if log_grep "END status=0" "$log_file"; then
    printf 'success'
    return 0
  fi
  if log_grep "END status=" "$log_file"; then
    printf 'failed'
    return 0
  fi
  printf 'running'
}

is_log_stale() {
  local log_file="$1"
  if [[ $STALE_HOURS -le 0 ]]; then
    return 1
  fi
  if [[ ! -f "$log_file" ]]; then
    return 1
  fi
  local now
  local mtime
  now="$(date +%s)"
  mtime="$(stat -c %Y "$log_file" 2>/dev/null || echo 0)"
  if [[ "$mtime" -eq 0 ]]; then
    return 1
  fi
  local age_hours=$(( (now - mtime) / 3600 ))
  if [[ $age_hours -ge $STALE_HOURS ]]; then
    return 0
  fi
  return 1
}

if [[ $DRY_RUN -eq 0 ]]; then
  init_gpu_queue
fi

EXPS=(centralized fedavg bank_perclient)
TASKS=(hellaswag piqa xsum hotpotqa mbpp apps toolbench)
if [[ $INCLUDE_GSM8K -eq 1 ]]; then
  TASKS+=(gsm8k)
fi
SKIP_TASKS=()
if [[ -n "$SKIP_TASKS_OVERRIDE" ]]; then
  IFS=',' read -r -a SKIP_TASKS <<< "${SKIP_TASKS_OVERRIDE// /}"
fi

is_skipped_task() {
  local task="$1"
  for skip in "${SKIP_TASKS[@]}"; do
    if [[ "$task" == "$skip" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "Results root not found: $RESULTS_ROOT" >&2
  exit 1
fi

RUN_IDS=()
if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_IDS+=("$RUN_ID_OVERRIDE")
else
  while IFS= read -r -d '' dir; do
    RUN_IDS+=("$(basename "$dir")")
  done < <(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

TOTAL_TASKS=0
SKIPPED_DONE=0
SKIPPED_RUNNING=0
SKIPPED_NOCKPT=0
SCHEDULED=0

TASK_QUEUE=()
QUEUE_DELIM=$'\t'

for run_id in "${RUN_IDS[@]}"; do
  run_root="$RESULTS_ROOT/$run_id"
  [[ -d "$run_root" ]] || continue

  if [[ -n "$RUN_TAGS_OVERRIDE" ]]; then
    IFS=',' read -r -a RUN_TAGS <<< "${RUN_TAGS_OVERRIDE// /}"
  else
    RUN_TAGS=()
    while IFS= read -r -d '' dir; do
      RUN_TAGS+=("$(basename "$dir")")
    done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  fi

  for run_tag in "${RUN_TAGS[@]}"; do
    run_dir="$run_root/$run_tag"
    [[ -d "$run_dir" ]] || continue
    for exp in "${EXPS[@]}"; do
      ckpt_path="$(find_ckpt "$run_dir" "$exp")"
      if [[ -z "$ckpt_path" ]]; then
        SKIPPED_NOCKPT=$((SKIPPED_NOCKPT + 1))
        continue
      fi
      for task in "${TASKS[@]}"; do
        if is_skipped_task "$task"; then
          continue
        fi
        if [[ "$task" == "toolbench" && $SKIP_TOOLBENCH_IF_UNSET -eq 1 ]]; then
          if [[ -z "${TOOLBENCH_EVAL_CMD:-}" || -z "${TOOLBENCH_EVAL_OUTPUT:-}" ]]; then
            continue
          fi
        fi
        TOTAL_TASKS=$((TOTAL_TASKS + 1))
        if find_accuracy "$run_dir" "$exp" "$task"; then
          SKIPPED_DONE=$((SKIPPED_DONE + 1))
          continue
        fi
        eval_id="eval_fanout_${run_id}_${run_tag}"
        log_dir="$LOGS_ROOT/$run_id/$run_tag"
        eval_log="$log_dir/eval_${exp}_${task}.log"
        state="$(eval_log_state "$eval_log")"
        if [[ "$state" == "running" ]]; then
          if [[ $FORCE_RERUN -eq 1 ]] || is_log_stale "$eval_log"; then
            :
          else
            SKIPPED_RUNNING=$((SKIPPED_RUNNING + 1))
            continue
          fi
        fi
        TASK_QUEUE+=("${run_id}${QUEUE_DELIM}${run_tag}${QUEUE_DELIM}${exp}${QUEUE_DELIM}${task}${QUEUE_DELIM}${ckpt_path}${QUEUE_DELIM}${eval_id}")
      done
    done
  done
done

log "Eval tasks total=$TOTAL_TASKS skip_done=$SKIPPED_DONE skip_running=$SKIPPED_RUNNING skip_nockpt=$SKIPPED_NOCKPT queued=${#TASK_QUEUE[@]}"

if [[ ${#TASK_QUEUE[@]} -eq 0 ]]; then
  log "No eval tasks to run."
  exit 0
fi

if (( SERVER_IDX < 0 || SERVER_IDX >= SERVER_COUNT )); then
  echo "Invalid server index $SERVER_IDX for server count $SERVER_COUNT" >&2
  exit 1
fi

for idx in "${!TASK_QUEUE[@]}"; do
  if (( idx % SERVER_COUNT != SERVER_IDX )); then
    continue
  fi
  IFS=$'\t' read -r run_id run_tag exp task ckpt_path eval_id <<< "${TASK_QUEUE[$idx]}"
  run_dir="$RESULTS_ROOT/$run_id/$run_tag"
  res_dir="$run_dir/global/$exp/$task/$eval_id"
  log_dir="$LOGS_ROOT/$run_id/$run_tag"
  mkdir -p "$res_dir" "$log_dir"
  eval_log="$log_dir/eval_${exp}_${task}.log"

  eval_yaml="$res_dir/eval_${exp}_${task}.yaml"
  max_new_tokens=64
  case "$task" in
    gsm8k)
      max_new_tokens=256
      ;;
    hellaswag|piqa)
      max_new_tokens=8
      ;;
    xsum)
      max_new_tokens=128
      ;;
    hotpotqa)
      max_new_tokens=32
      ;;
    mbpp|apps|toolbench)
      max_new_tokens=256
      ;;
  esac

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
  num_completions: 1
  timeout: 5
  max_samples_per_subject: 2
  superglue_tasks: ["boolq", "rte", "cb", "copa", "wic"]
EVAL_CFG
  sed -i \
    -e "s#__RES_DIR__#$res_dir#g" \
    -e "s#__CKPT__#$ckpt_path#g" \
    -e "s#__MAX_SAMPLES__#$EVAL_MAX_SAMPLES#g" \
    -e "s#__MAX_NEW_TOKENS__#$max_new_tokens#g" \
    "$eval_yaml"

  case "$task" in
    gsm8k)
      eval_cmd=(python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$eval_yaml")
      ;;
    hellaswag)
      eval_cmd=(python federatedscope/llm/eval/eval_for_hellaswag/eval.py --cfg "$eval_yaml")
      ;;
    piqa)
      eval_cmd=(python federatedscope/llm/eval/eval_for_piqa/eval.py --cfg "$eval_yaml")
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
    apps)
      eval_cmd=(python federatedscope/llm/eval/eval_for_apps/eval.py --cfg "$eval_yaml")
      ;;
    toolbench)
      eval_cmd=(python federatedscope/llm/eval/eval_for_toolbench/eval.py --cfg "$eval_yaml")
      ;;
    *)
      log "Unknown task: $task"
      continue
      ;;
  esac

  run_job "eval-${run_id}-${run_tag}-${exp}-${task}" "$eval_log" \
    bash -c "TMP_BASE='$run_dir/tmp'; \
      mkdir -p \"\$TMP_BASE\" '$res_dir/wandb'; \
      export TMPDIR=\"\$TMP_BASE/${eval_id}_${exp}_${task}\" WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1; \
      unset WANDB_SERVICE; \
      ${eval_cmd[*]}"
  SCHEDULED=$((SCHEDULED + 1))
done

log "Scheduled $SCHEDULED eval jobs on server ${SERVER_IDX}/${SERVER_COUNT}."
if ! wait_jobs; then
  log "Some eval jobs failed on server ${SERVER_IDX}/${SERVER_COUNT}."
  exit 1
fi
log "Eval fanout complete on server ${SERVER_IDX}/${SERVER_COUNT}."
