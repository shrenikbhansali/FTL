#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
SKIP_TRAIN=0
RESUME=0
GPUS="0,1,2,3"
PIPE_ID="${INDIVIDUAL_PIPE_ID:-}"
DATA_ROOT="${INDIVIDUAL_DATA_ROOT:-data/individual_federated}"
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-100}"
TRAIN_OPTS="${INDIVIDUAL_TRAIN_OPTS:-}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""
RUN_TAGS_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_individual_5multiconfig_pipeline.sh [options]

Options:
  --dry-run        Print commands without running them.
  --skip-train     Skip training jobs (eval only).
  --skip-eval      Skip evaluation jobs.
  --resume         Skip steps with existing outputs.
  --gpus           Comma-separated GPU list (default: 0,1,2,3).
  --pipe-id        Override pipeline id (default: auto).
  --run-tags       Comma-separated run tags to execute (default: all).
  --data-root      Dataset root (default: data/individual_federated).
  --eval-max-samples Max samples per eval task (default: 100).
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
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
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
  PIPE_ID="5multiconfig_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

BASE_DIR="$ROOT_DIR/5multiconfig"
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
TASKS=(hellaswag piqa xsum hotpotqa mbpp apps toolbench gsm8k)
TOTAL_FAILURES=0

RUN_TAGS=(participation_high participation_low more_shards drift_heavy balanced_shards)
declare -A CFG_PREFIXES=(
  [participation_high]="individual_federated_sharded_participation_high"
  [participation_low]="individual_federated_sharded_participation_low"
  [more_shards]="individual_federated_sharded_more_shards"
  [drift_heavy]="individual_federated_sharded_drift_heavy"
  [balanced_shards]="individual_federated_sharded_balanced_shards"
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

if [[ $SKIP_TRAIN -eq 0 ]]; then
  log "Launching 5multiconfig training jobs..."
  for run_tag in "${RUN_TAGS[@]}"; do
    cfg_prefix="${CFG_PREFIXES[$run_tag]:-}"
    if [[ -z "$cfg_prefix" ]]; then
      log "Unknown run tag: $run_tag"
      TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
      continue
    fi
    run_log_dir="$LOG_DIR/$run_tag"
    run_results_dir="$RESULTS_DIR/$run_tag"
    mkdir -p "$run_log_dir" "$run_results_dir"
    for exp in "${EXPS[@]}"; do
      OUTDIR="$run_results_dir/$exp/train"
      CKPT_PATH="$OUTDIR/ckpt.ckpt"
      TRAIN_LOG="$run_log_dir/train_${exp}.log"
      CFG="yamls/${cfg_prefix}_${exp}.yaml"
      if [[ $RESUME -eq 1 && -n "$(find_ckpt "$run_tag" "$exp")" ]]; then
        ts_now="$(ts)"
        record_status "train" "train-$run_tag-$exp" "-" "skipped" "$ts_now" "$ts_now" "$TRAIN_LOG"
        log "Skipping training for $run_tag/$exp (checkpoint exists)."
        continue
      fi
      mkdir -p "$OUTDIR"
      run_job "train" "train-$run_tag-$exp" "$TRAIN_LOG" \
        bash -c "export TMPDIR='$OUTDIR/tmp' WANDB_DIR='$OUTDIR/wandb' \
          WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
          mkdir -p '$OUTDIR/tmp' '$OUTDIR/wandb'; \
          unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
          python federatedscope/main.py \
            --cfg '$CFG' \
            outdir '$OUTDIR' expname '$exp' expname_tag 'run_${PIPE_ID}_${run_tag}' \
            federate.save_to '$CKPT_PATH' \
            ${TRAIN_OPT_ARGS[*]}"
    done
  done

  if ! wait_jobs "train"; then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
  fi
else
  log "Skipping training stage."
fi

if [[ $SKIP_EVAL -eq 0 ]]; then
  log "Launching evaluation jobs (gsm8k last)..."
  for run_tag in "${RUN_TAGS[@]}"; do
    run_log_dir="$LOG_DIR/$run_tag"
    run_results_dir="$RESULTS_DIR/$run_tag"
    mkdir -p "$run_log_dir" "$run_results_dir"
    eval_id="eval_${PIPE_ID}_${run_tag}"
    for exp in "${EXPS[@]}"; do
      ckpt_path="$(find_ckpt "$run_tag" "$exp")"
      if [[ -z "$ckpt_path" ]]; then
        log "No checkpoint found for $run_tag/$exp; skipping eval."
        continue
      fi
      for task in "${TASKS[@]}"; do
        res_dir="$run_results_dir/global/$exp/$task/$eval_id"
        eval_log="$run_log_dir/eval_${exp}_${task}.log"
        eval_yaml="$res_dir/eval_${exp}_${task}.yaml"
        if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
          ts_now="$(ts)"
          record_status "eval" "eval-$run_tag-$exp-$task" "-" "skipped" "$ts_now" "$ts_now" "$eval_log"
          log "Skipping eval for $run_tag/$exp/$task (results exist)."
          continue
        fi
        mkdir -p "$res_dir"
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
        run_job "eval" "eval-$run_tag-$exp-$task" "$eval_log" \
          bash -c "TMP_BASE='$run_results_dir/tmp'; \
            mkdir -p \"\$TMP_BASE\" '$res_dir/wandb'; \
            export TMPDIR=\"\$TMP_BASE/${eval_id}_${exp}_${task}\" WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1; \
            unset WANDB_SERVICE; \
            ${eval_cmd[*]}"
      done
    done
  done

  log "Waiting for evaluation jobs..."
  if ! wait_jobs "eval"; then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
  fi

  log "Collecting metrics per run..."
  for run_tag in "${RUN_TAGS[@]}"; do
    run_log_dir="$LOG_DIR/$run_tag"
    run_results_dir="$RESULTS_DIR/$run_tag"
    eval_id="eval_${PIPE_ID}_${run_tag}"
    for exp in "${EXPS[@]}"; do
      collect_log="$run_log_dir/collect_${exp}.log"
      : > "$collect_log"
      collect_cmd=(python scripts/collect_tulu_eval_results.py
        --results-root "$run_results_dir"
        --exp "$exp"
        --eval-job-id "$eval_id"
      )
      if [[ "$exp" == "bank_perclient" ]]; then
        collect_cmd+=(--fedavg-eval-job-id "$eval_id")
      fi
      log "Running metrics collection for $run_tag/$exp (log: $collect_log)"
      (export TMPDIR="$run_results_dir/tmp" WANDB_DIR="$run_results_dir/wandb" WANDB_DISABLE_SERVICE=1; \
        mkdir -p "$TMPDIR" "$WANDB_DIR"; \
        unset WANDB_SERVICE; \
        "${collect_cmd[@]}") >> "$collect_log" 2>&1
    done
  done
else
  log "Skipping evaluation stage."
fi

log "5multiconfig pipeline complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs executed."
fi
if [[ $TOTAL_FAILURES -ne 0 ]]; then
  log "Pipeline finished with failures. See $STATUS_FILE for details."
  exit 1
fi
