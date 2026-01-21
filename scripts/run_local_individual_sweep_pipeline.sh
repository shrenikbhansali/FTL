#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
SKIP_TRAIN=0
RESUME=0
GPUS="0,1,2,3"
SWEEP_ID="${INDIVIDUAL_SWEEP_ID:-}"
TRAIN_OPTS="${INDIVIDUAL_TRAIN_OPTS:-}"
DATA_ROOT="${INDIVIDUAL_DATA_ROOT:-data/individual_federated}"
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-50}"
RUN_TAGS_OVERRIDE=""
CONDA_ENV="${INDIVIDUAL_CONDA_ENV:-}"
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_individual_sweep_pipeline.sh [options]

Options:
  --dry-run          Print commands without running them.
  --skip-train       Skip training jobs and go straight to eval (if enabled).
  --skip-eval        Skip evaluation jobs.
  --resume           Skip steps with existing outputs.
  --gpus             Comma-separated GPU list (default: 0,1,2,3).
  --sweep-id         Override sweep id (default: auto).
  --train-opts       Extra federatedscope overrides (use :: as separator).
  --data-root        Dataset root (default: data/individual_federated).
  --eval-max-samples Max samples per eval task (default: 50).
  --run-tags         Comma-separated run tags to execute (default: all).
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
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
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
    --sweep-id)
      SWEEP_ID="$2"
      shift 2
      ;;
    --train-opts)
      TRAIN_OPTS="$2"
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
    --run-tags)
      RUN_TAGS_OVERRIDE="$2"
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

if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="individual_sweep_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

SWEEP_LOG_DIR="$ROOT_DIR/individual_sweep_logs/$SWEEP_ID"
PIPE_LOG="$SWEEP_LOG_DIR/pipeline.log"
STATUS_FILE="$SWEEP_LOG_DIR/status.tsv"
RUNS_FILE="$SWEEP_LOG_DIR/runs.tsv"
mkdir -p "$SWEEP_LOG_DIR"
: > "$PIPE_LOG"
printf "phase	run_tag	label	gpu	status	start	end	log
" > "$STATUS_FILE"
printf "run_tag	mode	run_id	cfg_prefix	script
" > "$RUNS_FILE"

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
  printf "[%s] %s
" "$(date +"%Y-%m-%dT%H:%M:%S%z")"     "conda not found; continuing without activation." >&2
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
JOB_PIPES=()

init_gpu_queue() {
  local fifo
  fifo="$(mktemp -u)"
  mkfifo "$fifo"
  exec {GPU_QUEUE_FD}<>"$fifo"
  rm -f "$fifo"
  for gpu in "${GPU_LIST[@]}"; do
    printf '%s
' "$gpu" >&"$GPU_QUEUE_FD"
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
  printf '%s
' "$1" >&"$GPU_QUEUE_FD"
}

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  printf "[%s] %s
" "$(ts)" "$*" | tee -a "$PIPE_LOG" >&2
}

declare -A RUN_STATUS_FILE

record_status() {
  local phase="$1"
  local run_tag="$2"
  local label="$3"
  local gpu="$4"
  local status="$5"
  local start_ts="$6"
  local end_ts="$7"
  local log_file="$8"
  printf "%s	%s	%s	%s	%s	%s	%s	%s
"     "$phase" "$run_tag" "$label" "$gpu" "$status"     "$start_ts" "$end_ts" "$log_file" >> "$STATUS_FILE"
  local run_status="${RUN_STATUS_FILE[$run_tag]:-}"
  if [[ -n "$run_status" ]]; then
    printf "%s	%s	%s	%s	%s	%s	%s
"       "$phase" "$label" "$gpu" "$status" "$start_ts" "$end_ts" "$log_file" >> "$run_status"
  fi
}

run_job() {
  local phase="$1"
  local run_tag="$2"
  local label="$3"
  local log_file="$4"
  shift 4

  local start_ts
  start_ts="$(ts)"
  if [[ $DRY_RUN -eq 1 ]]; then
    local gpu
    gpu="$(next_dry_gpu)"
    log "DRY-RUN [$phase][$run_tag] $label gpu=$gpu cmd: $*"
    record_status "$phase" "$run_tag" "$label" "$gpu"       "dry-run" "$start_ts" "$start_ts" "$log_file"
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
      printf "[%s] END status=%s
" "$(ts)" "$status" >> "$log_file"
      exit "$status"
    }
    trap cleanup EXIT
    printf "[%s] START gpu=%s
" "$(ts)" "$gpu" >> "$log_file"
    printf "[%s] CMD: %s
" "$(ts)" "$*" >> "$log_file"
    ensure_conda_libs
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
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
  JOB_PIPES+=("$run_tag")
}

remove_job_at_index() {
  local idx="$1"
  unset 'JOB_PIDS[idx]'
  unset 'JOB_LABELS[idx]'
  unset 'JOB_LOGS[idx]'
  unset 'JOB_GPUS[idx]'
  unset 'JOB_STARTS[idx]'
  unset 'JOB_PHASES[idx]'
  unset 'JOB_PIPES[idx]'
  JOB_PIDS=("${JOB_PIDS[@]}")
  JOB_LABELS=("${JOB_LABELS[@]}")
  JOB_LOGS=("${JOB_LOGS[@]}")
  JOB_GPUS=("${JOB_GPUS[@]}")
  JOB_STARTS=("${JOB_STARTS[@]}")
  JOB_PHASES=("${JOB_PHASES[@]}")
  JOB_PIPES=("${JOB_PIPES[@]}")
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
    local run_tag="${JOB_PIPES[$idx]}"
    local status=0
    if wait "$pid"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
    fi
    local end_ts
    end_ts="$(ts)"
    record_status "$phase" "$run_tag" "$label" "$gpu"       "$status" "$start_ts" "$end_ts" "$log_file"
    if [[ $status -ne 0 ]]; then
      log "Job failed: $label ($run_tag, gpu=$gpu, log=$log_file)"
    else
      log "Job finished: $label ($run_tag, gpu=$gpu)"
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
WAITED_JOB_PIPE=""
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
      local run_tag="${JOB_PIPES[$idx]}"
      local status=0
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      local end_ts
      end_ts="$(ts)"
      record_status "train" "$run_tag" "$label" "$gpu"         "$status" "$start_ts" "$end_ts" "$log_file"
      if [[ $status -ne 0 ]]; then
        log "Job failed: $label ($run_tag, gpu=$gpu, log=$log_file)"
      else
        log "Job finished: $label ($run_tag, gpu=$gpu)"
      fi
      WAITED_JOB_LABEL="$label"
      WAITED_JOB_STATUS="$status"
      WAITED_JOB_PIPE="$run_tag"
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

RUNS=(participation budget sharded_light setting2 setting3)
EXPS=(fedavg bank_perclient centralized)
TASKS=(gsm8k hellaswag piqa xsum hotpotqa mbpp apps toolbench)

declare -A RUN_MODE
RUN_MODE[participation]="normal"
RUN_MODE[budget]="normal"
RUN_MODE[sharded_light]="sharded"
RUN_MODE[setting2]="sharded"
RUN_MODE[setting3]="sharded"

declare -A RUN_CFG_PREFIX
RUN_CFG_PREFIX[participation]="individual_federated_participation"
RUN_CFG_PREFIX[budget]="individual_federated_budget"
RUN_CFG_PREFIX[sharded_light]="individual_federated_sharded_light"
RUN_CFG_PREFIX[setting2]="individual_federated_sharded_setting2"
RUN_CFG_PREFIX[setting3]="individual_federated_sharded_setting3"

declare -A RUN_ID
declare -A RUN_EVAL_ID
declare -A RUN_RESULT_DIR
declare -A RUN_LOG_DIR

set_run_paths() {
  local run_tag="$1"
  local run_id="${SWEEP_ID}_${run_tag}"
  RUN_ID[$run_tag]="$run_id"
  RUN_EVAL_ID[$run_tag]="eval_${run_id}"
  if [[ "${RUN_MODE[$run_tag]}" == "sharded" ]]; then
    RUN_RESULT_DIR[$run_tag]="$ROOT_DIR/individual_sharded_results/$run_id"
    RUN_LOG_DIR[$run_tag]="$ROOT_DIR/individual_sharded_logs/$run_id"
  else
    RUN_RESULT_DIR[$run_tag]="$ROOT_DIR/individual_results/$run_id"
    RUN_LOG_DIR[$run_tag]="$ROOT_DIR/individual_logs/$run_id"
  fi
}

if [[ -n "$RUN_TAGS_OVERRIDE" ]]; then
  IFS=',' read -r -a REQUESTED_RUNS <<< "${RUN_TAGS_OVERRIDE// /}"
  RUNS=()
  for tag in "${REQUESTED_RUNS[@]}"; do
    if [[ -z "${RUN_MODE[$tag]:-}" ]]; then
      echo "Unknown run tag: $tag" >&2
      exit 1
    fi
    RUNS+=("$tag")
  done
fi

for run_tag in "${RUNS[@]}"; do
  set_run_paths "$run_tag"
  mkdir -p "${RUN_LOG_DIR[$run_tag]}" "${RUN_RESULT_DIR[$run_tag]}"
  : > "${RUN_LOG_DIR[$run_tag]}/pipeline.log"
  RUN_STATUS_FILE[$run_tag]="${RUN_LOG_DIR[$run_tag]}/status.tsv"
  printf "phase	label	gpu	status	start	end	log
" > "${RUN_STATUS_FILE[$run_tag]}"
  printf "%s	%s	%s	%s	%s
"     "$run_tag" "${RUN_MODE[$run_tag]}" "${RUN_ID[$run_tag]}" "${RUN_CFG_PREFIX[$run_tag]}" "run_local_individual_sweep_pipeline.sh" >> "$RUNS_FILE"
  log "Initialized $run_tag run: results=${RUN_RESULT_DIR[$run_tag]}, logs=${RUN_LOG_DIR[$run_tag]}"
  log "  cfg_prefix=${RUN_CFG_PREFIX[$run_tag]} eval_id=${RUN_EVAL_ID[$run_tag]} mode=${RUN_MODE[$run_tag]}"
  log "  data_root=${DATA_ROOT} eval_max_samples=${EVAL_MAX_SAMPLES}"
  log "  run_id=${RUN_ID[$run_tag]}"
  log "---"
done

BASE_TRAIN_OPTS="data.tulu3_federated.root ${DATA_ROOT#data/}"
if [[ -n "$TRAIN_OPTS" ]]; then
  FULL_TRAIN_OPTS="${BASE_TRAIN_OPTS}::${TRAIN_OPTS}"
else
  FULL_TRAIN_OPTS="$BASE_TRAIN_OPTS"
fi
TRAIN_OPT_ARGS=()
read -r -a TRAIN_OPT_ARGS <<< "${FULL_TRAIN_OPTS//::/ }"

TOTAL_FAILURES=0
train_jobs=0


declare -A EVAL_SCHEDULED

enqueue_eval_jobs() {
  local run_tag="$1"
  local exp="$2"
  local key="$run_tag:$exp"
  if [[ -n "${EVAL_SCHEDULED[$key]:-}" ]]; then
    return 0
  fi
  local ckpt_path
  ckpt_path="$(find_ckpt "${RUN_RESULT_DIR[$run_tag]}" "$exp")"
  if [[ -z "$ckpt_path" ]]; then
    log "No checkpoint found for $run_tag/$exp; skipping eval."
    EVAL_SCHEDULED["$key"]=1
    return 0
  fi

  for task in "${TASKS[@]}"; do
    local res_dir="${RUN_RESULT_DIR[$run_tag]}/global/$exp/$task/${RUN_EVAL_ID[$run_tag]}"
    if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
      log "Skipping eval for $run_tag/$exp/$task (results exist)."
      local ts_now
      ts_now="$(ts)"
      record_status "eval" "$run_tag" "eval-$run_tag-$exp-$task" "-"         "skipped" "$ts_now" "$ts_now" "${RUN_LOG_DIR[$run_tag]}/eval_${exp}_${task}.log"
      continue
    fi
    local eval_log="${RUN_LOG_DIR[$run_tag]}/eval_${exp}_${task}.log"
    local eval_yaml="$res_dir/eval_${exp}_${task}.yaml"
    mkdir -p "$res_dir"
    local max_samples=200
    if [[ -n "$EVAL_MAX_SAMPLES" ]]; then
      max_samples="$EVAL_MAX_SAMPLES"
    fi
    local mmlu_max_samples_per_subject=2
    local max_new_tokens=64
    local num_completions=1
    local timeout_sec=5
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
      -e "s#__MMLU_SAMPLES__#$mmlu_max_samples_per_subject#g" \
      "$eval_yaml"
    local eval_cmd=()
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
    run_job "eval" "$run_tag" "eval-$run_tag-$exp-$task" "$eval_log"       bash -c "TMP_BASE='${RUN_RESULT_DIR[$run_tag]}/tmp';         mkdir -p "\$TMP_BASE" '$res_dir/wandb';         export TMPDIR="\$TMP_BASE/${RUN_EVAL_ID[$run_tag]}_${exp}_${task}" WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1;         unset WANDB_SERVICE;         ${eval_cmd[*]}"
  done
  EVAL_SCHEDULED["$key"]=1
}

if [[ $SKIP_TRAIN -eq 0 ]]; then
  log "Launching sweep training jobs..."
  for exp in "${EXPS[@]}"; do
    for run_tag in "${RUNS[@]}"; do
      local_result_dir="${RUN_RESULT_DIR[$run_tag]}"
      local_log_dir="${RUN_LOG_DIR[$run_tag]}"
      EXISTING_CKPT="$(find_ckpt "$local_result_dir" "$exp")"
      if [[ $RESUME -eq 1 && -n "$EXISTING_CKPT" ]]; then
        log "Skipping training for $run_tag/$exp (checkpoint exists)."
        ts_now="$(ts)"
        record_status "train" "$run_tag" "train-$run_tag-$exp" "-"           "skipped" "$ts_now" "$ts_now" "$local_log_dir/train_${exp}.log"
        if [[ $SKIP_EVAL -eq 0 ]]; then
          enqueue_eval_jobs "$run_tag" "$exp"
        fi
        continue
      fi
      OUTDIR="$local_result_dir/$exp/train"
      CKPT_PATH="$OUTDIR/ckpt.ckpt"
      mkdir -p "$OUTDIR"
      TRAIN_LOG="$local_log_dir/train_${exp}.log"
      CFG="yamls/${RUN_CFG_PREFIX[$run_tag]}_${exp}.yaml"
      run_job "train" "$run_tag" "train-$run_tag-$exp" "$TRAIN_LOG"         bash -c "export TMPDIR='$OUTDIR/tmp' WANDB_DIR='$OUTDIR/wandb'           WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1;           mkdir -p '$OUTDIR/tmp' '$OUTDIR/wandb';           unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID;           python federatedscope/main.py             --cfg '$CFG'             outdir '$OUTDIR' expname '$exp' expname_tag 'run_${SWEEP_ID}_${run_tag}'             federate.save_to '$CKPT_PATH'             ${TRAIN_OPT_ARGS[*]}"
      if [[ $DRY_RUN -eq 0 ]]; then
        train_jobs=$((train_jobs + 1))
      fi
    done
  done

  if [[ $SKIP_EVAL -eq 0 ]]; then
    while [[ $train_jobs -gt 0 ]]; do
      wait_for_any_train_job
      train_jobs=$((train_jobs - 1))
      if [[ $WAITED_JOB_STATUS -ne 0 ]]; then
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
      fi
      label="${WAITED_JOB_LABEL#train-}"
      run_tag="${label%%-*}"
      exp="${label#${run_tag}-}"
      enqueue_eval_jobs "$run_tag" "$exp"
    done
  else
    if ! wait_jobs "train"; then
      TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    fi
  fi
else
  log "Skipping training stage."
  if [[ $SKIP_EVAL -eq 0 ]]; then
    for exp in "${EXPS[@]}"; do
      for run_tag in "${RUNS[@]}"; do
        enqueue_eval_jobs "$run_tag" "$exp"
      done
    done
  fi
fi

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Waiting for evaluations..."
  if ! wait_jobs "eval"; then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
  fi

  log "Collecting metrics..."
  for run_tag in "${RUNS[@]}"; do
    for exp in "${EXPS[@]}"; do
      COLLECT_LOG="${RUN_LOG_DIR[$run_tag]}/collect_${exp}.log"
      if [[ $DRY_RUN -eq 1 ]]; then
        log "DRY-RUN [collect-$run_tag-$exp] python scripts/collect_tulu_eval_results.py --results-root ${RUN_RESULT_DIR[$run_tag]} --exp $exp --eval-job-id ${RUN_EVAL_ID[$run_tag]}"
        continue
      fi
      : > "$COLLECT_LOG"
      COLLECT_CMD=(python scripts/collect_tulu_eval_results.py
        --results-root "${RUN_RESULT_DIR[$run_tag]}"
        --exp "$exp"
        --eval-job-id "${RUN_EVAL_ID[$run_tag]}"
      )
      if [[ "$exp" == "bank_perclient" ]]; then
        COLLECT_CMD+=(--fedavg-eval-job-id "${RUN_EVAL_ID[$run_tag]}")
      fi
      log "Running metrics collection for $run_tag/$exp (log: $COLLECT_LOG)"
      (export TMPDIR="${RUN_RESULT_DIR[$run_tag]}/tmp" WANDB_DIR="${RUN_RESULT_DIR[$run_tag]}/wandb" WANDB_DISABLE_SERVICE=1;         mkdir -p "$TMPDIR" "$WANDB_DIR";         unset WANDB_SERVICE;         "${COLLECT_CMD[@]}") >> "$COLLECT_LOG" 2>&1
    done
  done
else
  log "Skipping evaluations."
fi

log "Hyperparam sweep complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs executed."
fi
if [[ $TOTAL_FAILURES -ne 0 ]]; then
  log "Sweep finished with failures. See $STATUS_FILE for details."
  exit 1
fi
