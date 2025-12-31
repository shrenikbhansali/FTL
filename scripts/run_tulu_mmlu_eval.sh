#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_tulu_mmlu_eval.sh [options]

Options:
  --dry-run   Print the sbatch commands without submitting.
  -h, --help  Show this message.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
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
  printf "[tulu-mmlu] %s\n" "$*" >&2
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

run mkdir -p logs results_tulu

submit_sbatch "mmlu-centralized" "$SBATCH_DIR/sbatch_eval_tulu_mmlu_centralized.sbatch"
submit_sbatch "mmlu-fedavg" "$SBATCH_DIR/sbatch_eval_tulu_mmlu_fedavg.sbatch"
submit_sbatch "mmlu-bank" "$SBATCH_DIR/sbatch_eval_tulu_mmlu_bank_perclient.sbatch"

log "Tulu MMLU submissions complete."
if [[ $DRY_RUN -eq 1 ]]; then
  log "Dry-run requested; no jobs were queued."
fi
