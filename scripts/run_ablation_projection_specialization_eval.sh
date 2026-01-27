#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_ablation_projection_specialization_eval.sh [options]

Evaluate specialization retention (per-client checkpoint on its own dataset)
for the ablation pair: server-only vs full (projection).

Options:
  --run-dir PATH           Base ablation results dir containing ablation runs.
                           Default: FTL/final/results/ablations/individual_ablations_20260122_191942_5628
  --gpus LIST              Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --eval-max-samples N     Limit eval samples per task (default: 100)
  --full-eval              Use full evaluation (clears --eval-max-samples)
  --pipe-id ID             Override pipeline id (default: auto)
  --resume                 Skip evals with existing results
  --hf-home PATH           Override HF_HOME
  -h, --help               Show this help
USAGE
}

RUN_DIR_DEFAULT="FTL/final/results/ablations/individual_ablations_20260122_191942_5628"
GPU_LIST="0,1,2,3,4,5,6,7"
EVAL_MAX_SAMPLES="100"
FULL_EVAL=0
PIPE_ID=""
RESUME=0
HF_HOME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR_DEFAULT="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
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

RUN_DIR="$RUN_DIR_DEFAULT"
if [[ "$RUN_DIR" == FTL/* ]]; then
  RUN_DIR="${RUN_DIR#FTL/}"
fi
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$ROOT_DIR/$RUN_DIR"
fi

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

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="spec_projection_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi
EVAL_ID="eval_${PIPE_ID}"

RESULTS_DIR="$ROOT_DIR/final/results/ablations/specialization_projection/$PIPE_ID"
LOG_DIR="$ROOT_DIR/final/logs/ablations/specialization_projection/$PIPE_ID"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export PYTHONUNBUFFERED=1

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$msg" | tee -a "$LOG_DIR/pipeline.log"
}

declare -A DATASET_CLIENT=(
  [gsm8k]="gsm8k_client"
  [hellaswag]="hellaswag_client"
  [xsum]="xsum_client"
  [hotpotqa]="hotpotqa_client"
  [mbpp]="mbpp_client"
)
DATASETS=(gsm8k hellaswag xsum hotpotqa mbpp)
EXPS=(abl_server_only abl_full)

MANIFEST_PATH="$ROOT_DIR/data/individual_federated/manifest.json"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  log "Manifest not found: $MANIFEST_PATH"
  exit 1
fi

declare -A CLIENT_IDS=()
while IFS='=' read -r name idx; do
  if [[ -n "$name" && -n "$idx" ]]; then
    CLIENT_IDS["$name"]="$idx"
  fi
done < <(python3 - "$MANIFEST_PATH" <<'PY'
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

max_new_tokens_for_task() {
  case "$1" in
    gsm8k) printf '256' ;;
    hellaswag) printf '8' ;;
    xsum) printf '128' ;;
    hotpotqa) printf '32' ;;
    mbpp) printf '256' ;;
    *) printf '64' ;;
  esac
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

start_eval_job() {
  local exp_tag="$1"
  local task="$2"
  local ckpt_path="$3"
  local gpu="$4"
  LAST_PID=""

  local res_dir="$RESULTS_DIR/spec/$exp_tag/$task/$EVAL_ID"
  local eval_log="$LOG_DIR/eval_${exp_tag}_${task}.log"

  if [[ $RESUME -eq 1 ]] && has_eval_result "$res_dir"; then
    log "Skipping eval for $exp_tag/$task (results exist)."
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

  log "Launching eval $exp_tag/$task on GPU $gpu"
  (
    TMP_BASE="$RESULTS_DIR/tmp"
    mkdir -p "$TMP_BASE" "$res_dir/wandb"
    export TMPDIR="$TMP_BASE/${EVAL_ID}_${exp_tag}_${task}"
    export WANDB_DIR="$res_dir/wandb"
    export WANDB_DISABLE_SERVICE=1
    export CUDA_VISIBLE_DEVICES="$gpu"
    unset WANDB_SERVICE
    "${eval_cmd[@]}"
  ) > "$eval_log" 2>&1 &
  LAST_PID=$!
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  log "No GPUs specified."
  exit 1
fi

JOBS=()
for exp in "${EXPS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    client_name="${DATASET_CLIENT[$dataset]}"
    client_id="${CLIENT_IDS[$client_name]:-}"
    if [[ -z "$client_id" ]]; then
      log "Client mapping not found for $dataset ($client_name); skipping."
      continue
    fi
    ckpt_path="$RUN_DIR/$exp/train/ckpt/clients/client_${client_id}/client_${client_id}.ckpt"
    if [[ ! -f "$ckpt_path" ]]; then
      log "Missing checkpoint for $exp/$dataset: $ckpt_path"
      continue
    fi
    JOBS+=("$exp|$dataset|$ckpt_path")
  done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
  log "No eval jobs found (missing checkpoints?)."
  exit 1
fi

log "Starting specialization evals for ablations: ${EXPS[*]}"
log "Run dir: $RUN_DIR"
log "Results dir: $RESULTS_DIR"
log "Log dir: $LOG_DIR"

job_index=0
total_jobs=${#JOBS[@]}
gpu_count=${#GPUS[@]}

while [[ $job_index -lt $total_jobs ]]; do
  pids=()
  for ((i=0; i<gpu_count && job_index<total_jobs; i++)); do
    IFS='|' read -r exp_tag task ckpt_path <<< "${JOBS[$job_index]}"
    start_eval_job "$exp_tag" "$task" "$ckpt_path" "${GPUS[$i]}"
    if [[ -n "${LAST_PID:-}" ]]; then
      pids+=("$LAST_PID")
    fi
    job_index=$((job_index + 1))
  done
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}"
  fi
done

log "Specialization projection evals complete."
