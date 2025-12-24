#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SBATCH_DIR="$ROOT_DIR/scripts"

log() {
  printf "[llama-gsm8k-humaneval] %s\n" "$*" >&2
}

submit_sbatch() {
  local label=$1
  shift
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "$label submitted: $job_id ($*)"
  echo "$job_id"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log "Missing required sbatch file: $1"
    exit 1
  fi
}

build_mod_array() {
  local max=$1
  local remainder=$2
  local mod_base=${3:-4}
  local -a indices=()
  for ((idx=0; idx<=max; idx++)); do
    if (( idx % mod_base == remainder )); then
      indices+=("$idx")
    fi
  done
  printf "%s" "$(IFS=,; echo "${indices[*]}")"
}

require_file "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"
require_file "$SBATCH_DIR/sbatch_eval_global_full.sbatch"
require_file "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"
require_file "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"
require_file "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

# Baseline variants: 10 configs * 4 tasks => indices 0..39
BASELINE_GSM8K="$(build_mod_array 39 1 4)"
BASELINE_HUMANEVAL="$(build_mod_array 39 3 4)"
BASELINE_ARRAY="${BASELINE_GSM8K},${BASELINE_HUMANEVAL}"
submit_sbatch "eval-baselines-gsm8k-humaneval" --array="$BASELINE_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"

# Standard FL (global): tasks 0..7 (per mapping in sbatch)
submit_sbatch "eval-global-gsm8k-humaneval" --array=1,3,5,7 \
  "$SBATCH_DIR/sbatch_eval_global_full.sbatch"

# Standard FL (clients): indices 0..23
CLIENT_GSM8K="$(build_mod_array 23 1 4)"
CLIENT_HUMANEVAL="$(build_mod_array 23 3 4)"
CLIENT_ARRAY="${CLIENT_GSM8K},${CLIENT_HUMANEVAL}"
submit_sbatch "eval-clients-gsm8k-humaneval" --array="$CLIENT_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"

# Unlearn (global): indices 0..15
UNLEARN_GLOBAL_GSM8K="$(build_mod_array 15 1 4)"
UNLEARN_GLOBAL_HUMANEVAL="$(build_mod_array 15 3 4)"
UNLEARN_GLOBAL_ARRAY="${UNLEARN_GLOBAL_GSM8K},${UNLEARN_GLOBAL_HUMANEVAL}"
submit_sbatch "eval-unlearn-global-gsm8k-humaneval" --array="$UNLEARN_GLOBAL_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"

# Unlearn (clients): indices 0..47
UNLEARN_CLIENT_GSM8K="$(build_mod_array 47 1 4)"
UNLEARN_CLIENT_HUMANEVAL="$(build_mod_array 47 3 4)"
UNLEARN_CLIENT_ARRAY="${UNLEARN_CLIENT_GSM8K},${UNLEARN_CLIENT_HUMANEVAL}"
submit_sbatch "eval-unlearn-clients-gsm8k-humaneval" --array="$UNLEARN_CLIENT_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

log "Submitted gsm8k/humaneval evaluation jobs for baselines, FL, and UNLEARN."
