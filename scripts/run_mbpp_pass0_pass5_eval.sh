#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_mbpp_pass0_pass5_eval.sh [options]

Re-run MBPP evals with updated prompting/extraction and record pass@5.
Also runs a 1-completion eval (pass@1) and writes a summary that includes
pass@0 (set to 0.0), pass@1, and pass@5.

Options:
  --part N/M            Split jobs across M servers, run shard N (0-indexed).
                        Default: 0/1
  --gpus LIST           Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --num-shots N         MBPP few-shot count from prompt split (default: 0)
  --mbpp-config NAME    MBPP config name (e.g., sanitized). Default: (HF default)
  --include-spec-proj   Include specialization-projection ablation evals.
  --dry-run             Print commands without running.
  --hf-home PATH        Override HF_HOME.
  -h, --help            Show this help.
USAGE
}

PART_SPEC="0/1"
GPU_LIST="0,1,2,3,4,5,6,7"
NUM_SHOTS=0
MBPP_CONFIG=""
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
    --num-shots)
      NUM_SHOTS="$2"
      shift 2
      ;;
    --mbpp-config)
      MBPP_CONFIG="$2"
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
if [[ -n "$NUM_SHOTS" && ! "$NUM_SHOTS" =~ ^[0-9]+$ ]]; then
  echo "Invalid --num-shots (must be integer)." >&2
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

TMP_BASE="$ROOT_DIR/final/data/eval_outputs_pass0_pass5"
LOG_BASE="$ROOT_DIR/final/logs/mbpp_pass0_pass5"
mkdir -p "$TMP_BASE" "$LOG_BASE"

echo "Selected ${#SELECTED[@]} MBPP evals for part $PART_SPEC using GPUs: ${GPUS[*]}"

get_outdir() {
  python3 - <<'PY' "$1"
import sys
path = sys.argv[1]
outdir = None
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip().startswith('outdir:'):
            val = line.split(':', 1)[1].strip().strip('"').strip("'")
            outdir = val
            break
if outdir:
    print(outdir)
PY
}

latest_json() {
  python3 - <<'PY' "$1"
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = list(root.rglob('accuracies_*__mbpp.json'))
if not files:
    sys.exit(1)
files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(files[0])
PY
}

write_summary() {
  python3 - <<'PY' "$1" "$2" "$3"
import json
import sys
from pathlib import Path
outdir = Path(sys.argv[1])
pass1_json = Path(sys.argv[2])
pass5_json = Path(sys.argv[3])
data1 = json.loads(pass1_json.read_text())
data5 = json.loads(pass5_json.read_text())
summary = {
    "pass@0": 0.0,
    "pass@1": data1.get("pass@1"),
    "pass@5": data5.get("pass@5"),
    "pass@10": data5.get("pass@10"),
    "total_examples": data5.get("total_examples", data1.get("total_examples")),
    "pass1_source": str(pass1_json),
    "pass5_source": str(pass5_json),
}
out_path = outdir / "mbpp_pass0_pass5_summary.json"
out_path.write_text(json.dumps(summary, indent=2))
print(out_path)
PY
}

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
    log="$LOG_BASE/${safe%.yaml}.pass0_pass5.log"

    outdir="$(get_outdir "$yaml")"
    if [[ -z "$outdir" ]]; then
      echo "WARN: could not parse outdir for $yaml" >&2
      job_index=$((job_index + 1))
      continue
    fi

    base_cmd=(python federatedscope/llm/eval/eval_for_mbpp/eval.py --cfg "$yaml" \
      eval.mbpp_num_shots "$NUM_SHOTS" eval.mbpp_use_chat_prompt True)
    if [[ -n "$MBPP_CONFIG" ]]; then
      base_cmd+=(eval.mbpp_config "$MBPP_CONFIG")
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "CUDA_VISIBLE_DEVICES=$gpu ${base_cmd[*]} eval.num_completions 1 > $log 2>&1"
      echo "CUDA_VISIBLE_DEVICES=$gpu ${base_cmd[*]} eval.num_completions 5 >> $log 2>&1"
    else
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export TMPDIR="$TMP_BASE/mbpp_pass0_pass5_${gpu}_$RANDOM"
        mkdir -p "$TMPDIR"
        "${base_cmd[@]}" eval.num_completions 1 >> "$log" 2>&1
        pass1_json="$(latest_json "$outdir")"
        cp "$pass1_json" "$outdir/mbpp_pass1_latest.json"
        "${base_cmd[@]}" eval.num_completions 5 >> "$log" 2>&1
        pass5_json="$(latest_json "$outdir")"
        cp "$pass5_json" "$outdir/mbpp_pass5_latest.json"
        write_summary "$outdir" "$pass1_json" "$pass5_json" >> "$log" 2>&1
      ) &
      pids+=("$!")
    fi
    job_index=$((job_index + 1))
  done
  if [[ $DRY_RUN -eq 0 && ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}"
  fi
done

echo "MBPP pass0/pass5 evals complete for part $PART_SPEC."
