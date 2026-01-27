#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_mbpp_pass5_eval.sh [options]

Re-run MBPP evals with pass@5 by setting eval.num_completions=5 for all
paper-related experiments (baseline, centralized single-task, specialization
retention, and ablations).

Options:
  --part N/M            Split jobs across M servers, run shard N (0-indexed).
                        Default: 0/1
  --gpus LIST           Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --include-spec-proj   Include specialization-projection ablation evals.
  --dry-run             Print commands without running.
  --hf-home PATH        Override HF_HOME.
  -h, --help            Show this help.
USAGE
}

PART_SPEC="0/1"
GPU_LIST="0,1,2,3,4,5,6,7"
INCLUDE_SPEC_PROJ=0
DRY_RUN=0
HF_HOME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --part)
      PART_SPEC="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --include-spec-proj)
      INCLUDE_SPEC_PROJ=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ "$PART_SPEC" != */* ]]; then
  echo "Invalid --part (expected N/M)." >&2
  exit 1
fi
PART_IDX="${PART_SPEC%%/*}"
PARTS="${PART_SPEC#*/}"
if [[ -z "$PART_IDX" || -z "$PARTS" ]]; then
  echo "Invalid --part (expected N/M)." >&2
  exit 1
fi
if ! [[ "$PART_IDX" =~ ^[0-9]+$ && "$PARTS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --part (expected integers)." >&2
  exit 1
fi
if [[ "$PARTS" -le 0 || "$PART_IDX" -ge "$PARTS" ]]; then
  echo "Invalid --part (N must be < M, M > 0)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/heck2/sbhansali8/HFcache"
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLE_SERVICE=1

YAMLS=(
  "$ROOT_DIR/individual_results/individual_local_20260115_100457_21987/global/centralized/mbpp/eval_individual_local_20260115_100457_21987/eval_centralized_mbpp.yaml"
  "$ROOT_DIR/individual_results/individual_local_20260115_100457_21987/global/fedavg/mbpp/eval_individual_local_20260115_100457_21987/eval_fedavg_mbpp.yaml"
  "$ROOT_DIR/individual_results/individual_local_20260115_100457_21987/global/bank_perclient/mbpp/eval_individual_local_20260115_100457_21987/eval_bank_perclient_mbpp.yaml"
  "$ROOT_DIR/final/results/single_client/global/centralized_single_mbpp/mbpp/eval_individual_single_task_20260122_001626_21663/eval_centralized_single_mbpp_mbpp.yaml"
  "$ROOT_DIR/individual_results/individual_specialization_20260121_182914_18926/global/spec_fedavg_mbpp/mbpp/eval_individual_specialization_20260121_182914_18926/eval_spec_fedavg_mbpp_mbpp.yaml"
  "$ROOT_DIR/individual_results/individual_specialization_20260121_182914_18926/global/spec_bank_perclient_mbpp/mbpp/eval_individual_specialization_20260121_182914_18926/eval_spec_bank_perclient_mbpp_mbpp.yaml"
)

while IFS= read -r -d '' f; do
  YAMLS+=("$f")
done < <(find "$ROOT_DIR/final/results/ablations/individual_ablations_20260122_191942_5628/global" \
  -path "*/mbpp/*yaml" -print0 | sort -z)

if [[ $INCLUDE_SPEC_PROJ -eq 1 ]]; then
  while IFS= read -r -d '' f; do
    YAMLS+=("$f")
  done < <(find "$ROOT_DIR/final/results/ablations/specialization_projection" \
    -path "*/mbpp/*yaml" -print0 | sort -z)
fi

FILTERED=()
for f in "${YAMLS[@]}"; do
  if [[ -f "$f" ]]; then
    FILTERED+=("$f")
  else
    echo "WARN: missing eval yaml: $f" >&2
  fi
done

if [[ ${#FILTERED[@]} -eq 0 ]]; then
  echo "No eval YAMLs found." >&2
  exit 1
fi

IFS=$'\n' read -r -d '' -a FILTERED_SORTED < <(printf '%s\n' "${FILTERED[@]}" | sort -u && printf '\0')

SELECTED=()
for i in "${!FILTERED_SORTED[@]}"; do
  if (( i % PARTS == PART_IDX )); then
    SELECTED+=("${FILTERED_SORTED[$i]}")
  fi
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "No jobs selected for part $PART_SPEC." >&2
  exit 1
fi

TMP_BASE="$ROOT_DIR/final/data/eval_outputs_pass5"
LOG_BASE="$ROOT_DIR/final/logs/mbpp_pass5"
mkdir -p "$TMP_BASE" "$LOG_BASE"

echo "Selected ${#SELECTED[@]} MBPP evals for part $PART_SPEC using GPUs: ${GPUS[*]}"

job_index=0
total_jobs=${#SELECTED[@]}
gpu_count=${#GPUS[@]}

while [[ $job_index -lt $total_jobs ]]; do
  pids=()
  for ((i=0; i<gpu_count && job_index<total_jobs; i++)); do
    yaml="${SELECTED[$job_index]}"
    gpu="${GPUS[$i]}"
    rel="${yaml#$ROOT_DIR/}"
    safe="${rel//\//_}"
    log="$LOG_BASE/${safe%.yaml}.pass5.log"

    cmd=(python federatedscope/llm/eval/eval_for_mbpp/eval.py --cfg "$yaml" eval.num_completions 5)
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]} > $log 2>&1"
    else
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TMPDIR="$TMP_BASE/mbpp_pass5_${gpu}_$RANDOM"
        mkdir -p "$TMPDIR"
        "${cmd[@]}"
      ) > "$log" 2>&1 &
      pids+=("$!")
    fi
    job_index=$((job_index + 1))
  done
  if [[ $DRY_RUN -eq 0 && ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}"
  fi
done

echo "MBPP pass@5 evals complete for part $PART_SPEC."
