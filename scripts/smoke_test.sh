#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_BASELINES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-baselines)
      SKIP_BASELINES=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--dry-run] [--skip-baselines]" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SBATCH_DIR="$ROOT_DIR/scripts"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Run this script on a Slurm login node." >&2
  exit 1
fi

log() {
  printf "[smoke] %s\n" "$*" >&2
}

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN: $*"
  else
    "$@"
  fi
}

ensure_data_links() {
  local -a stems=(
    alpaca_data
    chat_dolly
    code_alpaca
    composite_llm3
    math_gsm8k
    rosetta_alpaca
  )
  for stem in "${stems[@]}"; do
    local target="data/${stem}.json"
    if [[ -f "$target" || -L "$target" ]]; then
      continue
    fi
    local source="data/${stem}"
    if [[ -f "$source" ]]; then
      log "Creating symlink $target -> $source"
      run ln -s "$source" "$target"
    else
      log "Skipping missing dataset stem: $source"
    fi
  done
}

submit_sbatch() {
  local label=$1
  shift
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN sbatch ($label): $*"
    echo "dry-$label"
    return
  fi
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "$label job submitted: $job_id"
  echo "$job_id"
}

ensure_data_links
run mkdir -p logs results

log "Submitting smoke-test training jobs..."
LLAMA_JOB=$(submit_sbatch "train-llama" "$SBATCH_DIR/sbatch_train_llama.sbatch")
QWEN_JOB=$(submit_sbatch "train-qwen" "$SBATCH_DIR/sbatch_train_qwen_moe.sbatch")

if [[ $SKIP_BASELINES -eq 0 ]]; then
  BASELINE_JOB=$(submit_sbatch "train-baselines" "$SBATCH_DIR/sbatch_train_baselines.sbatch")
fi

log "Submitting global evaluation arrays..."
submit_sbatch "eval-global-llama" --dependency=afterok:${LLAMA_JOB} --array=0-3 "$SBATCH_DIR/sbatch_eval_global.sbatch"
submit_sbatch "eval-global-qwen" --dependency=afterok:${QWEN_JOB} --array=4-7 "$SBATCH_DIR/sbatch_eval_global.sbatch"

log "Submitting per-client evaluation arrays..."
submit_sbatch "eval-clients-llama" --dependency=afterok:${LLAMA_JOB} --array=0-11 "$SBATCH_DIR/sbatch_eval_clients.sbatch"
submit_sbatch "eval-clients-qwen" --dependency=afterok:${QWEN_JOB} --array=12-23 "$SBATCH_DIR/sbatch_eval_clients.sbatch"

if [[ $SKIP_BASELINES -eq 0 ]]; then
  log "Submitting baseline evaluations..."
  submit_sbatch "eval-baselines" --dependency=afterok:${BASELINE_JOB} "$SBATCH_DIR/sbatch_eval_baselines.sbatch"
fi

log "Smoke-test submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
