#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SBATCH_DIR="$ROOT_DIR/scripts"

log() {
  printf "[llama-gsm8k-code] %s\n" "$*" >&2
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
  local -a indices=()
  for ((idx=0; idx<=max; idx++)); do
    if (( idx % 3 == remainder )); then
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

# Baseline variants: 10 configs * 3 tasks => indices 0..29
BASELINE_GSM8K="$(build_mod_array 29 1)"
BASELINE_CODE="$(build_mod_array 29 2)"
BASELINE_ARRAY="${BASELINE_GSM8K},${BASELINE_CODE}"
submit_sbatch "eval-baselines-gsm8k-code" --array="$BASELINE_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_baselines_full.sbatch"

# Standard FL (global): tasks 0..5 (per mapping in sbatch)
submit_sbatch "eval-global-gsm8k-code" --array=1,2,4,5 \
  "$SBATCH_DIR/sbatch_eval_global_full.sbatch"

# Standard FL (clients): indices 0..17
CLIENT_GSM8K="$(build_mod_array 17 1)"
CLIENT_CODE="$(build_mod_array 17 2)"
CLIENT_ARRAY="${CLIENT_GSM8K},${CLIENT_CODE}"
submit_sbatch "eval-clients-gsm8k-code" --array="$CLIENT_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_clients_full.sbatch"

# Unlearn (global): indices 0..11
UNLEARN_GLOBAL_GSM8K="$(build_mod_array 11 1)"
UNLEARN_GLOBAL_CODE="$(build_mod_array 11 2)"
UNLEARN_GLOBAL_ARRAY="${UNLEARN_GLOBAL_GSM8K},${UNLEARN_GLOBAL_CODE}"
submit_sbatch "eval-unlearn-global-gsm8k-code" --array="$UNLEARN_GLOBAL_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_unlearn_global_full.sbatch"

# Unlearn (clients): indices 0..35
UNLEARN_CLIENT_GSM8K="$(build_mod_array 35 1)"
UNLEARN_CLIENT_CODE="$(build_mod_array 35 2)"
UNLEARN_CLIENT_ARRAY="${UNLEARN_CLIENT_GSM8K},${UNLEARN_CLIENT_CODE}"
submit_sbatch "eval-unlearn-clients-gsm8k-code" --array="$UNLEARN_CLIENT_ARRAY" \
  "$SBATCH_DIR/sbatch_eval_unlearn_clients_full.sbatch"

log "Submitted gsm8k/code evaluation jobs for baselines, FL, and UNLEARN."
