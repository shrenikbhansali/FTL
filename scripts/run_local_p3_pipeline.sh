#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
SKIP_TRAIN=0
RESUME=0
GPUS="0,1,2,3"
VARIANTS="dataset"
PIPE_ID="${P3_PIPE_ID:-}"
EVAL_ID="${P3_EVAL_ID:-}"
TRAIN_OPTS="${P3_TRAIN_OPTS:-}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_p3_pipeline.sh [options]

Options:
  --dry-run        Print commands without running them.
  --skip-train     Skip training jobs and go straight to eval (if enabled).
  --skip-eval      Skip evaluation jobs.
  --resume         Skip steps with existing outputs.
  --gpus           Comma-separated GPU list (default: 0,1,2,3).
  --variant        Single variant: config, dataset, or category.
  --variants       Comma-separated variants or "all".
  --pipe-id        Override pipeline id base (default: auto).
  --eval-id        Override evaluation id base (default: eval_<pipe-id>).
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
    --variant|--variants)
      VARIANTS="$2"
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
  PIPE_ID="p3_local_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  if [[ -n "${PIPE_LOG:-}" ]]; then
    printf "[%s] %s\n" "$(ts)" "$*" | tee -a "$PIPE_LOG" >&2
  else
    printf "[%s] %s\n" "$(ts)" "$*" >&2
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
  log "conda not found; continuing without activation."
}

maybe_activate_conda "$CONDA_ENV"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export PYTHONUNBUFFERED=1
# Ensure conda libstdc++ is preferred over the system copy for SciPy/sklearn.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

GPUS="${GPUS// /}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

VARIANTS="${VARIANTS// /}"
VARIANTS="${VARIANTS,,}"
if [[ "$VARIANTS" == "all" ]]; then
  VARIANTS="config,dataset,category"
fi
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
if [[ ${#VARIANT_LIST[@]} -eq 0 ]]; then
  echo "No P3 variants provided." >&2
  exit 1
fi
for variant in "${VARIANT_LIST[@]}"; do
  case "$variant" in
    config|dataset|category)
      ;;
    *)
      echo "Unknown P3 variant: $variant" >&2
      exit 1
      ;;
  esac
done

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

WAITED_JOB_LABEL=""
WAITED_JOB_STATUS=0
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
      local status=0
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      local end_ts
      end_ts="$(ts)"
      record_status "train" "$label" "$gpu" "$status" "$start_ts" "$end_ts" "$log_file"
      if [[ $status -ne 0 ]]; then
        log "Job failed: $label (gpu=$gpu, log=$log_file)"
      else
        log "Job finished: $label (gpu=$gpu)"
      fi
      WAITED_JOB_LABEL="$label"
      WAITED_JOB_STATUS="$status"
      WAITED_JOB_GPU="$gpu"
      WAITED_JOB_LOG="$log_file"
      remove_job_at_index "$idx"
      return 0
    done
    sleep 1
  done
}

find_ckpt() {
  local exp="$1"
  local base="$RESULTS_DIR/$exp/train"
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

EXPS=(fedavg bank_perclient centralized)
TASKS=(mmlu gsm8k humaneval superglue)
TOTAL_FAILURES=0
PIPE_LOG=""
STATUS_FILE=""
declare -A EVAL_SCHEDULED=()

enqueue_eval_jobs() {
  local exp="$1"
  if [[ -n "${EVAL_SCHEDULED[$exp]:-}" ]]; then
    return 0
  fi
  local ckpt_path
  ckpt_path="$(find_ckpt "$exp")"
  if [[ -z "$ckpt_path" ]]; then
    log "No checkpoint found for $exp; skipping eval."
    EVAL_SCHEDULED["$exp"]=1
    return 0
  fi
  for task in "${TASKS[@]}"; do
    local res_dir="$RESULTS_DIR/global/$exp/$task/$EVAL_ID_VAR"
    if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
      log "Skipping eval for $exp/$task (results exist)."
      local ts_now
      ts_now="$(ts)"
      record_status "eval" "eval-$exp-$task" "-" "skipped" "$ts_now" "$ts_now" "$LOG_DIR/eval_${exp}_${task}.log"
      continue
    fi
    mkdir -p "$res_dir"
    local eval_log="$LOG_DIR/eval_${exp}_${task}.log"
    local eval_yaml="$res_dir/eval_${exp}_${task}.yaml"
    local max_samples=200
    local mmlu_max_samples_per_subject=2
    cat > "$eval_yaml" <<EOF
use_gpu: True
device: 0
outdir: "$res_dir/exp"
federate:
  save_to: "$ckpt_path"
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
  max_samples: $max_samples
  max_samples_per_subject: $mmlu_max_samples_per_subject
  superglue_tasks: ["boolq", "rte", "cb", "copa", "wic"]
EOF
    local eval_cmd=()
    case "$task" in
      mmlu)
        eval_cmd=(python federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg "$eval_yaml")
        ;;
      gsm8k)
        eval_cmd=(python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$eval_yaml")
        ;;
      humaneval)
        eval_cmd=(python federatedscope/llm/eval/eval_for_humaneval/eval.py --cfg "$eval_yaml")
        ;;
      superglue)
        eval_cmd=(python federatedscope/llm/eval/eval_for_superglue/eval.py --cfg "$eval_yaml")
        ;;
      *)
        log "Unknown task: $task"
        continue
        ;;
    esac
    run_job "eval" "eval-$exp-$task" "$eval_log" \
      bash -c "TMP_BASE='$RESULTS_DIR/tmp'; \
        mkdir -p \"\$TMP_BASE\" '$res_dir/wandb'; \
        export TMPDIR=\"\$TMP_BASE/${EVAL_ID_VAR}_${exp}_${task}\" WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1; \
        unset WANDB_SERVICE; \
        ${eval_cmd[*]}"
  done
  EVAL_SCHEDULED["$exp"]=1
}

for variant in "${VARIANT_LIST[@]}"; do
  if [[ ${#VARIANT_LIST[@]} -gt 1 ]]; then
    PIPE_ID_VAR="${PIPE_ID}_${variant}"
  else
    PIPE_ID_VAR="$PIPE_ID"
  fi
  if [[ -n "$EVAL_ID" ]]; then
    if [[ ${#VARIANT_LIST[@]} -gt 1 ]]; then
      EVAL_ID_VAR="${EVAL_ID}_${variant}"
    else
      EVAL_ID_VAR="$EVAL_ID"
    fi
  else
    EVAL_ID_VAR="eval_${PIPE_ID_VAR}"
  fi

  LOG_DIR="$ROOT_DIR/p3_logs/$variant/$PIPE_ID_VAR"
  RESULTS_DIR="$ROOT_DIR/p3_results/$variant/$PIPE_ID_VAR"
  PIPE_LOG="$LOG_DIR/pipeline.log"
  STATUS_FILE="$LOG_DIR/status.tsv"
  EVAL_SCHEDULED=()
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"
  : > "$PIPE_LOG"
  printf "phase\tlabel\tgpu\tstatus\tstart\tend\tlog\n" > "$STATUS_FILE"

  DATA_ROOT="data/p3_federated_${variant}"
  GROUP_BY="$variant"
  CATEGORY_MAP="$ROOT_DIR/materials/p3_category_map.json"

  export P3_PIPE_ID="$PIPE_ID_VAR"
  export P3_RESULTS_DIR="$RESULTS_DIR"
  export P3_LOGS_DIR="$LOG_DIR"
  export P3_DATA_ROOT="$DATA_ROOT"
  export P3_GROUP_BY="$GROUP_BY"
  export P3_CATEGORY_MAP="$CATEGORY_MAP"

  BASE_TRAIN_OPTS="data.tulu3_federated.root ${DATA_ROOT#data/}"
  if [[ -n "$TRAIN_OPTS" ]]; then
    FULL_TRAIN_OPTS="${BASE_TRAIN_OPTS}::${TRAIN_OPTS}"
  else
    FULL_TRAIN_OPTS="$BASE_TRAIN_OPTS"
  fi
  TRAIN_OPT_ARGS=()
  read -r -a TRAIN_OPT_ARGS <<< "${FULL_TRAIN_OPTS//::/ }"

  FAILURES=0
  train_jobs=0
  if [[ $SKIP_TRAIN -eq 0 ]]; then
    log "Launching P3 $variant training jobs..."
    for exp in "${EXPS[@]}"; do
      EXISTING_CKPT="$(find_ckpt "$exp")"
      if [[ $RESUME -eq 1 && -n "$EXISTING_CKPT" ]]; then
        log "Skipping training for $exp (checkpoint exists)."
        ts_now="$(ts)"
        record_status "train" "train-$exp" "-" "skipped" "$ts_now" "$ts_now" "$LOG_DIR/train_${exp}.log"
        if [[ $SKIP_EVAL -eq 0 ]]; then
          enqueue_eval_jobs "$exp"
        fi
        continue
      fi
      OUTDIR="$RESULTS_DIR/$exp/train"
      CKPT_PATH="$OUTDIR/ckpt.ckpt"
      mkdir -p "$OUTDIR"
      TRAIN_LOG="$LOG_DIR/train_${exp}.log"
      CFG="yamls/p3_federated_${exp}.yaml"
      run_job "train" "train-$exp" "$TRAIN_LOG" \
        bash -c "export TMPDIR='$OUTDIR/tmp' WANDB_DIR='$OUTDIR/wandb' \
          WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
          mkdir -p '$OUTDIR/tmp' '$OUTDIR/wandb'; \
          unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
          python federatedscope/main.py \
            --cfg '$CFG' \
            outdir '$OUTDIR' expname '$exp' expname_tag 'run_${PIPE_ID_VAR}' \
            federate.save_to '$CKPT_PATH' \
            ${TRAIN_OPT_ARGS[*]}"
      if [[ $DRY_RUN -eq 0 ]]; then
        train_jobs=$((train_jobs + 1))
      fi
    done

    if [[ $SKIP_EVAL -eq 0 ]]; then
      while [[ $train_jobs -gt 0 ]]; do
        wait_for_any_train_job
        train_jobs=$((train_jobs - 1))
        if [[ $WAITED_JOB_STATUS -ne 0 ]]; then
          FAILURES=$((FAILURES + 1))
        fi
        exp="${WAITED_JOB_LABEL#train-}"
        enqueue_eval_jobs "$exp"
      done
    else
      if ! wait_jobs "train"; then
        FAILURES=$((FAILURES + 1))
      fi
    fi
  else
    log "Skipping training stage."
    if [[ $SKIP_EVAL -eq 0 ]]; then
      for exp in "${EXPS[@]}"; do
        enqueue_eval_jobs "$exp"
      done
    fi
  fi

  if [[ $SKIP_EVAL -eq 0 ]]; then
    log "Waiting for P3 $variant evaluations..."
    if ! wait_jobs "eval"; then
      FAILURES=$((FAILURES + 1))
    fi

    log "Collecting P3 $variant metrics..."
    for exp in "${EXPS[@]}"; do
      COLLECT_LOG="$LOG_DIR/collect_${exp}.log"
      if [[ $DRY_RUN -eq 1 ]]; then
        log "DRY-RUN [collect-$exp] python scripts/collect_tulu_eval_results.py --results-root $RESULTS_DIR --exp $exp --eval-job-id $EVAL_ID_VAR"
        continue
      fi
      : > "$COLLECT_LOG"
      COLLECT_CMD=(python scripts/collect_tulu_eval_results.py
        --results-root "$RESULTS_DIR"
        --exp "$exp"
        --eval-job-id "$EVAL_ID_VAR"
      )
      if [[ "$exp" == "bank_perclient" ]]; then
        COLLECT_CMD+=(--fedavg-eval-job-id "$EVAL_ID_VAR")
      fi
      log "Running metrics collection for $exp (log: $COLLECT_LOG)"
      (export TMPDIR="$RESULTS_DIR/tmp" WANDB_DIR="$RESULTS_DIR/wandb" WANDB_DISABLE_SERVICE=1; \
        mkdir -p "$TMPDIR" "$WANDB_DIR"; \
        unset WANDB_SERVICE; \
        "${COLLECT_CMD[@]}") >> "$COLLECT_LOG" 2>&1
    done
  else
    log "Skipping evaluations."
  fi

  log "P3 $variant local pipeline complete."
  if [[ $DRY_RUN -eq 1 ]]; then
    log "Dry-run requested; no jobs executed."
  fi
  if [[ $FAILURES -ne 0 ]]; then
    log "Pipeline finished with failures. See $STATUS_FILE for details."
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
  fi
done

if [[ $TOTAL_FAILURES -ne 0 ]]; then
  exit 1
fi
