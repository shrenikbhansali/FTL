#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
GPUS="0,1,2,3,4,5,6,7"
SERVER_COUNT=1
SERVER_IDX=0
RUN_ID=""
RUN_TAGS=""
INCLUDE_GSM8K=0
EVAL_MAX_SAMPLES="${INDIVIDUAL_EVAL_MAX_SAMPLES:-100}"
CONDA_ENV=""
HF_HOME_OVERRIDE=""
SKIP_TASKS=""
FORCE=0
STALE_HOURS=0

usage() {
  cat <<'USAGE'
Usage: run_local_individual_taskaware_5run_eval_a40.sh [options]

Options:
  --dry-run          Print planned jobs without running them.
  --gpus             Comma-separated GPU list (default: 0-7).
  --server-count     Total number of servers (default: 1).
  --server-idx       This server index [0..server-count-1] (default: 0).
  --run-id           5run pipeline id (default: latest in 5multiconfig/results).
  --run-tags         Comma-separated run tags (default: all taskaware tags).
  --include-gsm8k    Include GSM8K evals (default: off).
  --skip-tasks       Comma-separated tasks to skip (e.g. piqa,apps).
  --eval-max-samples Max samples per eval task (default: 100).
  --conda-env        Conda env to activate before running.
  --hf-home          Override HF_HOME for dataset/model cache.
  --resume           Skip evals with existing results.
  --force            Re-run even if eval log looks running/failed.
  --stale-hours      Consider running logs older than N hours as stale.
  -h, --help         Show this message.
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
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --server-count)
      SERVER_COUNT="$2"
      shift 2
      ;;
    --server-idx)
      SERVER_IDX="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --run-tags)
      RUN_TAGS="$2"
      shift 2
      ;;
    --include-gsm8k)
      INCLUDE_GSM8K=1
      shift
      ;;
    --skip-tasks)
      SKIP_TASKS="$2"
      shift 2
      ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
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
    --force)
      FORCE=1
      shift
      ;;
    --stale-hours)
      STALE_HOURS="$2"
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
RESULTS_ROOT="$ROOT_DIR/5multiconfig/results"
LOGS_ROOT="$ROOT_DIR/5multiconfig/logs"
FANOUT_SCRIPT="$ROOT_DIR/scripts/run_5multiconfig_eval_fanout.sh"

if [[ -z "$RUN_ID" ]]; then
  if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "Results root not found: $RESULTS_ROOT" >&2
    exit 1
  fi
  RUN_ID="$(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
  if [[ -z "$RUN_ID" ]]; then
    echo "No runs found in $RESULTS_ROOT" >&2
    exit 1
  fi
fi

CMD=(bash "$FANOUT_SCRIPT"
  --run-id "$RUN_ID"
  --results-root "$RESULTS_ROOT"
  --logs-root "$LOGS_ROOT"
  --gpus "$GPUS"
  --server-count "$SERVER_COUNT"
  --server-idx "$SERVER_IDX"
  --eval-max-samples "$EVAL_MAX_SAMPLES"
  --conda-env "$CONDA_ENV"
)

if [[ -n "$RUN_TAGS" ]]; then
  CMD+=(--run-tags "$RUN_TAGS")
else
  CMD+=(--run-tags "ta_rho05,ta_rho0,ta_rglobal2,ta_participation_mid,ta_beta50")
fi
if [[ $RESUME -eq 1 ]]; then
  CMD+=(--resume)
fi
if [[ $INCLUDE_GSM8K -eq 1 ]]; then
  CMD+=(--include-gsm8k)
fi
if [[ -n "$SKIP_TASKS" ]]; then
  CMD+=(--skip-tasks "$SKIP_TASKS")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=(--dry-run)
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi
if [[ $FORCE -eq 1 ]]; then
  CMD+=(--force)
fi
if [[ $STALE_HOURS -gt 0 ]]; then
  CMD+=(--stale-hours "$STALE_HOURS")
fi

"${CMD[@]}"
