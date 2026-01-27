#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
SKIP_SPEC_EVAL=0
SKIP_CEILING_TRAIN=0
SKIP_CEILING_EVAL=0
SKIP_COLLECT=0
RERUN_BASELINE_IF_MISSING=1
GPUS="0,1,2,3,4,5,6,7"
PIPE_ID="${INDIVIDUAL_SPECIALIZATION_PIPE_ID:-}"
EVAL_ID="${INDIVIDUAL_SPECIALIZATION_EVAL_ID:-}"
BASELINE_DIR="${INDIVIDUAL_BASELINE_DIR:-individual_results/individual_local_20260115_100457_21987}"
DATA_ROOT="${INDIVIDUAL_DATA_ROOT:-data/individual_federated}"
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-}"
CFG_PREFIX="${INDIVIDUAL_CFG_PREFIX:-individual_federated}"
CENTRALIZED_CFG="${INDIVIDUAL_CENTRALIZED_CFG:-yamls/individual_federated_centralized.yaml}"
TRAIN_OPTS="${INDIVIDUAL_TRAIN_OPTS:-}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""
FULL_EVAL=0

usage() {
  cat <<'USAGE'
Usage: run_local_individual_specialization_pipeline.sh [options]

Options:
  --dry-run              Print commands without running them.
  --resume               Skip steps with existing outputs.
  --skip-spec-eval        Skip specialization retention evals.
  --skip-ceiling-train    Skip centralized single-task training.
  --skip-ceiling-eval     Skip centralized single-task evals.
  --skip-collect          Skip metrics aggregation.
  --no-rerun-baseline     Do not rerun baseline training if client ckpts are missing.
  --gpus                  Comma-separated GPU list (default: 0,1,2,3,4,5,6,7).
  --pipe-id               Override pipeline id (default: auto).
  --eval-id               Override evaluation id base (default: eval_<pipe-id>).
  --baseline-dir          Baseline results dir for per-client ckpts.
  --data-root             Dataset root (default: data/individual_federated).
  --eval-max-samples      Max samples per eval task (omit for full eval).
  --full-eval             Run full benchmark (ignore eval-max-samples).
  --cfg-prefix            YAML prefix for baseline reruns (default: individual_federated).
  --centralized-cfg        Centralized YAML path (default: yamls/individual_federated_centralized.yaml).
  --train-opts            Extra federatedscope overrides (use :: as separator).
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
    --skip-spec-eval)
      SKIP_SPEC_EVAL=1
      shift
      ;;
    --skip-ceiling-train)
      SKIP_CEILING_TRAIN=1
      shift
      ;;
    --skip-ceiling-eval)
      SKIP_CEILING_EVAL=1
      shift
      ;;
    --skip-collect)
      SKIP_COLLECT=1
      shift
      ;;
    --no-rerun-baseline)
      RERUN_BASELINE_IF_MISSING=0
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
    --cfg-prefix)
      CFG_PREFIX="$2"
      shift 2
      ;;
    --centralized-cfg)
      CENTRALIZED_CFG="$2"
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

if [[ "$BASELINE_DIR" == FTL/* ]]; then
  BASELINE_DIR="${BASELINE_DIR#FTL/}"
fi
if [[ "$BASELINE_DIR" != /* ]]; then
  BASELINE_DIR="$ROOT_DIR/$BASELINE_DIR"
fi

DATA_ROOT_PATH="$DATA_ROOT"
if [[ "$DATA_ROOT_PATH" == FTL/* ]]; then
  DATA_ROOT_PATH="${DATA_ROOT_PATH#FTL/}"
fi
if [[ "$DATA_ROOT_PATH" != /* ]]; then
  DATA_ROOT_PATH="$ROOT_DIR/$DATA_ROOT_PATH"
fi

DATA_ROOT_SUFFIX="$DATA_ROOT"
if [[ "$DATA_ROOT_SUFFIX" == FTL/* ]]; then
  DATA_ROOT_SUFFIX="${DATA_ROOT_SUFFIX#FTL/}"
fi
if [[ "$DATA_ROOT_SUFFIX" == data/* ]]; then
  DATA_ROOT_SUFFIX="${DATA_ROOT_SUFFIX#data/}"
fi

if [[ "$CENTRALIZED_CFG" == FTL/* ]]; then
  CENTRALIZED_CFG="${CENTRALIZED_CFG#FTL/}"
fi
if [[ "$CENTRALIZED_CFG" != /* ]]; then
  CENTRALIZED_CFG="$ROOT_DIR/$CENTRALIZED_CFG"
fi

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="individual_specialization_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="eval_${PIPE_ID}"
fi

LOG_DIR="$ROOT_DIR/individual_logs/$PIPE_ID"
RESULTS_DIR="$ROOT_DIR/individual_results/$PIPE_ID"
PIPE_LOG="$LOG_DIR/pipeline.log"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
: > "$PIPE_LOG"
printf "phase\tlabel\tgpu\tstatus\tstart\tend\tlog\n" > "$STATUS_FILE"

export INDIVIDUAL_SPECIALIZATION_PIPE_ID="$PIPE_ID"
export INDIVIDUAL_SPECIALIZATION_RESULTS_DIR="$RESULTS_DIR"
export INDIVIDUAL_SPECIALIZATION_LOGS_DIR="$LOG_DIR"

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

client_ckpt_path() {
  local base_dir="$1"
  local exp="$2"
  local client_id="$3"
  printf '%s' "$base_dir/$exp/train/ckpt/clients/client_${client_id}/client_${client_id}.ckpt"
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
  if [[ -n "$max_samples" ]]; then
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
  else
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
  max_new_tokens: __MAX_NEW_TOKENS__
  num_completions: __NUM_COMPLETIONS__
  timeout: __TIMEOUT__
  max_samples_per_subject: __MMLU_SAMPLES__
  superglue_tasks: ["boolq", "rte", "cb", "copa", "wic"]
EVAL_CFG
  fi
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
  local exp_tag="$1"
  local task="$2"
  local ckpt_path="$3"

  if [[ ! -f "$ckpt_path" ]]; then
    log "No checkpoint for $exp_tag/$task: $ckpt_path"
    return 0
  fi

  local res_dir="$RESULTS_DIR/global/$exp_tag/$task/$EVAL_ID"
  local eval_log="$LOG_DIR/eval_${exp_tag}_${task}.log"

  if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
    log "Skipping eval for $exp_tag/$task (results exist)."
    local ts_now
    ts_now="$(ts)"
    record_status "eval" "eval-$exp_tag-$task" "-" "skipped" "$ts_now" "$ts_now" "$eval_log"
    return 0
  fi

  mkdir -p "$res_dir"
  local eval_yaml="$res_dir/eval_${exp_tag}_${task}.yaml"
  local max_samples="$EVAL_MAX_SAMPLES"
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

  run_job "eval" "eval-$exp_tag-$task" "$eval_log" \
    bash -c "TMP_BASE='$RESULTS_DIR/tmp'; \
      mkdir -p \"\$TMP_BASE\" '$res_dir/wandb'; \
      export TMPDIR=\"\$TMP_BASE/${EVAL_ID}_${exp_tag}_${task}\" \
        WANDB_DIR='$res_dir/wandb' WANDB_DISABLE_SERVICE=1; \
      unset WANDB_SERVICE; \
      ${eval_cmd[*]}"
}

declare -A DATASET_CLIENT=(
  [gsm8k]="gsm8k_client"
  [hellaswag]="hellaswag_client"
  [xsum]="xsum_client"
  [hotpotqa]="hotpotqa_client"
  [mbpp]="mbpp_client"
)
DATASETS=(gsm8k hellaswag xsum hotpotqa mbpp)

MANIFEST_PATH="$DATA_ROOT_PATH/manifest.json"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  log "Manifest not found: $MANIFEST_PATH"
  exit 1
fi

if [[ ! -f "$CENTRALIZED_CFG" ]]; then
  log "Centralized config not found: $CENTRALIZED_CFG"
  exit 1
fi

declare -A CLIENT_IDS=()
while IFS='=' read -r name idx; do
  if [[ -n "$name" && -n "$idx" ]]; then
    CLIENT_IDS["$name"]="$idx"
  fi
done < <(python - "$MANIFEST_PATH" <<'PY'
import json
import sys

manifest = sys.argv[1]
with open(manifest, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, client in enumerate(data.get("clients", []), start=1):
    name = client.get("name")
    if name:
        print(f"{name}={i}")
PY
)

for dataset in "${DATASETS[@]}"; do
  client_name="${DATASET_CLIENT[$dataset]:-}"
  if [[ -z "$client_name" ]]; then
    log "Missing client mapping for dataset: $dataset"
    exit 1
  fi
  if [[ -z "${CLIENT_IDS[$client_name]:-}" ]]; then
    log "Client not found in manifest: $client_name (dataset $dataset)"
    exit 1
  fi
done

if [[ $DRY_RUN -eq 0 ]]; then
  init_gpu_queue
fi

BASE_TRAIN_ARGS=(data.tulu3_federated.root "$DATA_ROOT_SUFFIX")
TRAIN_OPT_ARGS=()
if [[ -n "$TRAIN_OPTS" ]]; then
  read -r -a TRAIN_OPT_ARGS <<< "${TRAIN_OPTS//::/ }"
fi

spec_missing_clients() {
  local base_dir="$1"
  local exp="$2"
  local missing=()
  for dataset in "${DATASETS[@]}"; do
    local client_name="${DATASET_CLIENT[$dataset]}"
    local client_id="${CLIENT_IDS[$client_name]}"
    local ckpt_path="$base_dir/$exp/train/ckpt/clients/client_${client_id}/client_${client_id}.ckpt"
    if [[ ! -f "$ckpt_path" ]]; then
      missing+=("$dataset")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${missing[@]}"
  return 1
}

EXPS=(fedavg bank_perclient)
BASELINE_RERUN_DIR="$RESULTS_DIR/baseline_rerun"
declare -A BASELINE_DIR_BY_EXP=()
declare -A BASELINE_RERUN=()
declare -A BASELINE_MISSING_BY_EXP=()
declare -A COLLECT_EXPS=()

for exp in "${EXPS[@]}"; do
  if missing_list=$(spec_missing_clients "$BASELINE_DIR" "$exp"); then
    BASELINE_DIR_BY_EXP["$exp"]="$BASELINE_DIR"
  else
    missing_csv=$(echo "$missing_list" | paste -sd, -)
    BASELINE_DIR_BY_EXP["$exp"]="$BASELINE_DIR"
    BASELINE_MISSING_BY_EXP["$exp"]="$missing_list"
    if [[ $RERUN_BASELINE_IF_MISSING -eq 1 ]]; then
      log "Missing per-client ckpts for $exp under $BASELINE_DIR ($missing_csv); will rerun."
      BASELINE_RERUN["$exp"]=1
    else
      log "Missing per-client ckpts for $exp under $BASELINE_DIR ($missing_csv); skipping specialization eval for $exp."
    fi
  fi
done

FAILURES=0
baseline_train_jobs=0
if [[ $SKIP_SPEC_EVAL -eq 0 ]]; then
  log "Launching specialization retention evals (pre-training)..."
  for exp in "${EXPS[@]}"; do
    base_dir="${BASELINE_DIR_BY_EXP[$exp]:-}"
    if [[ -z "$base_dir" ]]; then
      log "Skipping specialization eval for $exp (no baseline dir)."
      continue
    fi
    for dataset in "${DATASETS[@]}"; do
      client_name="${DATASET_CLIENT[$dataset]}"
      client_id="${CLIENT_IDS[$client_name]}"
      ckpt_path="$(client_ckpt_path "$base_dir" "$exp" "$client_id")"
      if [[ ! -f "$ckpt_path" ]]; then
        log "Missing ckpt for $exp/$dataset (client $client_id): $ckpt_path"
        continue
      fi
      exp_tag="spec_${exp}_${dataset}"
      schedule_eval "$exp_tag" "$dataset" "$ckpt_path"
      COLLECT_EXPS["$exp_tag"]=1
    done
  done

  log "Waiting for specialization retention evals (pre-training)..."
  if ! wait_jobs "eval"; then
    FAILURES=$((FAILURES + 1))
  fi
fi

if [[ $RERUN_BASELINE_IF_MISSING -eq 1 ]]; then
  for exp in "${EXPS[@]}"; do
    if [[ -z "${BASELINE_RERUN[$exp]:-}" ]]; then
      continue
    fi
    if [[ $RESUME -eq 1 ]]; then
      if spec_missing_clients "$BASELINE_RERUN_DIR" "$exp" >/dev/null; then
        log "Skipping baseline rerun for $exp (per-client ckpts exist)."
        ts_now="$(ts)"
        record_status "train" "train-baseline-$exp" "-" "skipped" "$ts_now" "$ts_now" \
          "$LOG_DIR/train_baseline_${exp}.log"
        continue
      fi
    fi
    OUTDIR="$BASELINE_RERUN_DIR/$exp/train"
    CKPT_PATH="$OUTDIR/ckpt.ckpt"
    mkdir -p "$OUTDIR"
    TRAIN_LOG="$LOG_DIR/train_baseline_${exp}.log"
    CFG="$ROOT_DIR/yamls/${CFG_PREFIX}_${exp}.yaml"
    if [[ ! -f "$CFG" ]]; then
      log "Baseline config not found: $CFG"
      FAILURES=$((FAILURES + 1))
      continue
    fi
    run_job "train" "train-baseline-$exp" "$TRAIN_LOG" \
      bash -c "export TMPDIR='$OUTDIR/tmp' WANDB_DIR='$OUTDIR/wandb' \
        WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
        mkdir -p '$OUTDIR/tmp' '$OUTDIR/wandb'; \
        unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
        python federatedscope/main.py \
          --cfg '$CFG' \
          outdir '$OUTDIR' expname '$exp' expname_tag 'run_${PIPE_ID}' \
          federate.save_to '$CKPT_PATH' \
          ${BASE_TRAIN_ARGS[*]} ${TRAIN_OPT_ARGS[*]}"
    if [[ $DRY_RUN -eq 0 ]]; then
      baseline_train_jobs=$((baseline_train_jobs + 1))
    fi
  done

  if [[ $baseline_train_jobs -gt 0 ]]; then
    log "Waiting for baseline rerun training jobs..."
    if ! wait_jobs "train"; then
      FAILURES=$((FAILURES + 1))
    fi
  fi
fi

if [[ $SKIP_SPEC_EVAL -eq 0 ]]; then
  spec_rerun_jobs=0
  for exp in "${EXPS[@]}"; do
    if [[ -z "${BASELINE_RERUN[$exp]:-}" ]]; then
      continue
    fi
    missing_list="${BASELINE_MISSING_BY_EXP[$exp]:-}"
    if [[ -z "$missing_list" ]]; then
      continue
    fi
    while IFS= read -r dataset; do
      [[ -z "$dataset" ]] && continue
      client_name="${DATASET_CLIENT[$dataset]}"
      client_id="${CLIENT_IDS[$client_name]}"
      ckpt_path="$(client_ckpt_path "$BASELINE_RERUN_DIR" "$exp" "$client_id")"
      if [[ ! -f "$ckpt_path" ]]; then
        log "Missing rerun ckpt for $exp/$dataset (client $client_id): $ckpt_path"
        continue
      fi
      exp_tag="spec_${exp}_${dataset}"
      schedule_eval "$exp_tag" "$dataset" "$ckpt_path"
      COLLECT_EXPS["$exp_tag"]=1
      spec_rerun_jobs=$((spec_rerun_jobs + 1))
    done <<< "$missing_list"
  done

  if [[ $spec_rerun_jobs -gt 0 ]]; then
    log "Waiting for specialization retention evals (post-rerun)..."
    if ! wait_jobs "eval"; then
      FAILURES=$((FAILURES + 1))
    fi
  fi
fi

declare -A CENTRALIZED_TRAIN_DIR=()
centralized_train_jobs=0
if [[ $SKIP_CEILING_TRAIN -eq 0 ]]; then
  log "Launching centralized single-task training jobs..."
  for dataset in "${DATASETS[@]}"; do
    client_name="${DATASET_CLIENT[$dataset]}"
    outdir="$RESULTS_DIR/centralized_single/$dataset/train"
    CENTRALIZED_TRAIN_DIR["$dataset"]="$outdir"
    if [[ $RESUME -eq 1 ]]; then
      ckpt_existing="$(find_ckpt_in_dir "$outdir")"
      if [[ -n "$ckpt_existing" ]]; then
        log "Skipping training for $dataset (checkpoint exists)."
        ts_now="$(ts)"
        record_status "train" "train-centralized-$dataset" "-" "skipped" \
          "$ts_now" "$ts_now" "$LOG_DIR/train_centralized_${dataset}.log"
        continue
      fi
    fi
    mkdir -p "$outdir"
    ckpt_path="$outdir/ckpt.ckpt"
    TRAIN_LOG="$LOG_DIR/train_centralized_${dataset}.log"
    client_list=$(printf '["%s"]' "$client_name")
    run_job "train" "train-centralized-$dataset" "$TRAIN_LOG" \
      bash -c "export TMPDIR='$outdir/tmp' WANDB_DIR='$outdir/wandb' \
        WANDB_DISABLE_SERVICE=1 WANDB_USE=0 WANDB_DISABLED=1; \
        mkdir -p '$outdir/tmp' '$outdir/wandb'; \
        unset WANDB_SERVICE WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG WANDB_RUN_ID; \
        python federatedscope/main.py \
          --cfg '$CENTRALIZED_CFG' \
          outdir '$outdir' expname 'centralized_${dataset}' expname_tag 'run_${PIPE_ID}' \
          federate.save_to '$ckpt_path' \
          ${BASE_TRAIN_ARGS[*]} ${TRAIN_OPT_ARGS[*]} \
          data.tulu3_federated.clients '$client_list'"
    if [[ $DRY_RUN -eq 0 ]]; then
      centralized_train_jobs=$((centralized_train_jobs + 1))
    fi
  done

  if [[ $centralized_train_jobs -gt 0 ]]; then
    log "Waiting for centralized single-task training jobs..."
    if ! wait_jobs "train"; then
      FAILURES=$((FAILURES + 1))
    fi
  fi
else
  for dataset in "${DATASETS[@]}"; do
    CENTRALIZED_TRAIN_DIR["$dataset"]="$RESULTS_DIR/centralized_single/$dataset/train"
  done
fi

if [[ $SKIP_CEILING_EVAL -eq 0 ]]; then
  log "Launching centralized single-task evals..."
  for dataset in "${DATASETS[@]}"; do
    train_dir="${CENTRALIZED_TRAIN_DIR[$dataset]}"
    ckpt_path="$(find_ckpt_in_dir "$train_dir")"
    if [[ -z "$ckpt_path" ]]; then
      log "No centralized checkpoint found for $dataset under $train_dir"
      continue
    fi
    exp_tag="centralized_single_${dataset}"
    schedule_eval "$exp_tag" "$dataset" "$ckpt_path"
    COLLECT_EXPS["$exp_tag"]=1
  done

  log "Waiting for centralized single-task evals..."
  if ! wait_jobs "eval"; then
    FAILURES=$((FAILURES + 1))
  fi
fi

if [[ $SKIP_COLLECT -eq 0 && ${#COLLECT_EXPS[@]} -gt 0 ]]; then
  log "Collecting eval metrics..."
  for exp_tag in "${!COLLECT_EXPS[@]}"; do
    COLLECT_LOG="$LOG_DIR/collect_${exp_tag}.log"
    if [[ $DRY_RUN -eq 1 ]]; then
      log "DRY-RUN [collect-$exp_tag] python scripts/collect_tulu_eval_results.py --results-root $RESULTS_DIR --exp $exp_tag --eval-job-id $EVAL_ID"
      continue
    fi
    : > "$COLLECT_LOG"
    log "Running metrics collection for $exp_tag (log: $COLLECT_LOG)"
    (export TMPDIR="$RESULTS_DIR/tmp" WANDB_DIR="$RESULTS_DIR/wandb" WANDB_DISABLE_SERVICE=1; \
      mkdir -p "$TMPDIR" "$WANDB_DIR"; \
      unset WANDB_SERVICE; \
      python scripts/collect_tulu_eval_results.py \
        --results-root "$RESULTS_DIR" \
        --exp "$exp_tag" \
        --eval-job-id "$EVAL_ID") >> "$COLLECT_LOG" 2>&1
  done
fi

log "Individual specialization/ceiling pipeline complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs executed."
fi
if [[ $FAILURES -ne 0 ]]; then
  log "Pipeline finished with failures. See $STATUS_FILE for details."
  exit 1
fi
