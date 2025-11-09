#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_GLOBAL=0
SKIP_CLIENTS=0
SKIP_BASELINES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-global)
      SKIP_GLOBAL=1
      shift
      ;;
    --skip-clients)
      SKIP_CLIENTS=1
      shift
      ;;
    --skip-baselines)
      SKIP_BASELINES=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--dry-run] [--skip-global] [--skip-clients] [--skip-baselines]" >&2
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
  printf "[eval-smoke] %s\n" "$*" >&2
}

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN: $*"
  else
    "$@"
  fi
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

run mkdir -p logs results

if [[ $SKIP_GLOBAL -eq 0 ]]; then
  # Global sbatch index mapping: 0=mmlu,1=gsm8k,2=code llama2; 5=code qwen_moe
  log "Submitting global code evaluation tasks..."
  submit_sbatch "eval-global-code" --array=2,5 "$SBATCH_DIR/sbatch_eval_global.sbatch"
else
  log "Skipping global evaluation submission."
fi

if [[ $SKIP_CLIENTS -eq 0 ]]; then
  # Client sbatch indices with code task: every third entry (2,5,8,11,14,17)
  log "Submitting client code evaluation tasks..."
  submit_sbatch "eval-clients-code" --array=2,5,8,11,14,17 "$SBATCH_DIR/sbatch_eval_clients.sbatch"
else
  log "Skipping client evaluation submission."
fi

if [[ $SKIP_BASELINES -eq 0 ]]; then
  # Baseline sbatch indices with code task: 2,5,8,11,14,17,20,23
  log "Submitting baseline code evaluation tasks..."
  submit_sbatch "eval-baselines-code" --array=2,5,8,11,14,17,20,23 "$SBATCH_DIR/sbatch_eval_baselines.sbatch"
else
  log "Skipping baseline evaluation submission."
fi

log "Eval smoke-test submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
