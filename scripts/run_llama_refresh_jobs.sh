#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SBATCH_DIR="$ROOT_DIR/scripts"

log() {
  printf "[llama-refresh] %s\n" "$*" >&2
}

submit_sbatch() {
  local label=$1
  shift
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "$label submitted: $job_id ($*)"
  echo "$job_id"
}

maybe_require() {
  if [[ ! -f "$1" ]]; then
    log "Missing required file: $1"
    exit 1
  fi
}

maybe_require "$SBATCH_DIR/sbatch_train_baselines_full.sbatch"
maybe_require "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"
maybe_require "$SBATCH_DIR/sbatch_eval_global_full.sbatch"
maybe_require "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"
maybe_require "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"
maybe_require "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

declare -A BASELINE_CKPTS=(
  [0]="ckpts/full/baselines/llama2_code_local.ckpt"
  [1]="ckpts/full/baselines/llama2_gsm8k_local.ckpt"
  [2]="ckpts/full/baselines/llama2_instr_local.ckpt"
)

declare -A INTERLEAVED_CKPTS=(
  [4]="ckpts/full/baselines/llama2_interleaved_all_in_one.ckpt"
)

missing_baseline_indices=()
for idx in "${!BASELINE_CKPTS[@]}"; do
  if [[ ! -f "${BASELINE_CKPTS[$idx]}" ]]; then
    missing_baseline_indices+=("$idx")
  fi
done

missing_interleaved_indices=()
for idx in "${!INTERLEAVED_CKPTS[@]}"; do
  if [[ ! -f "${INTERLEAVED_CKPTS[$idx]}" ]]; then
    missing_interleaved_indices+=("$idx")
  fi
done

BASELINE_JOB=""
if ((${#missing_baseline_indices[@]} > 0)); then
  BASELINE_JOB=$(submit_sbatch "train-llama-baselines" --array="$(IFS=,; echo "${missing_baseline_indices[*]}")" "$SBATCH_DIR/sbatch_train_baselines_full.sbatch")
else
  log "Dataset-specific baseline checkpoints already exist; skipping training."
fi

INTERLEAVED_JOB=""
if ((${#missing_interleaved_indices[@]} > 0)); then
  INTERLEAVED_JOB=$(submit_sbatch "train-llama-baseline-interleaved" --array="${missing_interleaved_indices[0]}" "$SBATCH_DIR/sbatch_train_baselines_full.sbatch")
else
  log "Interleaved baseline checkpoint already exists; skipping training."
fi

dependency_flags=()
dep_ids=()
[[ -n "$BASELINE_JOB" ]] && dep_ids+=("$BASELINE_JOB")
[[ -n "$INTERLEAVED_JOB" ]] && dep_ids+=("$INTERLEAVED_JOB")
if ((${#dep_ids[@]} > 0)); then
  dependency_flags=(--dependency=afterok:"$(IFS=:; echo "${dep_ids[*]}")")
fi

submit_sbatch "eval-baselines-llama" "${dependency_flags[@]}" --array=0-19 "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"
submit_sbatch "eval-global-llama" --array=0-3 "$SBATCH_DIR/sbatch_eval_global_full.sbatch"
submit_sbatch "eval-clients-llama" --array=0-11 "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"
submit_sbatch "eval-unlearn-global-llama" --array=0-7 "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"
submit_sbatch "eval-unlearn-clients-llama" --array=0-23 "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

log "Refresh submissions complete."
