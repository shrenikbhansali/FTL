#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
GPUS="0"
PIPE_ID="${INDIVIDUAL_ABLATION_SANITY_ID:-}"
RUN_TAG="${INDIVIDUAL_ABLATION_SANITY_TAG:-abl_proj_rank_max_16}"
RESULTS_ROOT="${INDIVIDUAL_ABLATION_SANITY_RESULTS_ROOT:-final/results/ablations_sanity}"
LOGS_ROOT="${INDIVIDUAL_ABLATION_SANITY_LOGS_ROOT:-final/logs/ablations_sanity}"
TRAIN_ROUNDS="${INDIVIDUAL_ABLATION_SANITY_ROUNDS:-1}"
LOCAL_STEPS="${INDIVIDUAL_ABLATION_SANITY_LOCAL_STEPS:-5}"
CONDA_ENV="${INDIVIDUAL_ABLATION_SANITY_CONDA_ENV:-}"
HF_HOME_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: run_local_individual_ablation_sanity.sh [options]

Runs a short ablation training job to validate A40 memory headroom.
Defaults to the most memory-intensive ablation config (abl_proj_rank_max_16)
with 1 round and 5 local steps.

Options:
  --dry-run          Print the pipeline command without running it.
  --gpus             Comma-separated GPU list (default: 0).
  --pipe-id          Override pipeline id (default: auto).
  --run-tag          Ablation tag to test (default: abl_proj_rank_max_16).
  --results-root     Results root (default: final/results/ablations_sanity).
  --logs-root        Logs root (default: final/logs/ablations_sanity).
  --rounds           Federated rounds (default: 1).
  --local-steps      Local update steps per round (default: 5).
  --conda-env        Conda env to activate before running.
  --hf-home          Override HF_HOME for dataset/model cache.
  -h, --help         Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --pipe-id)
      PIPE_ID="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --logs-root)
      LOGS_ROOT="$2"
      shift 2
      ;;
    --rounds)
      TRAIN_ROUNDS="$2"
      shift 2
      ;;
    --local-steps)
      LOCAL_STEPS="$2"
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

if [[ -z "$PIPE_ID" ]]; then
  PIPE_ID="individual_ablation_sanity_$(date +%Y%m%d_%H%M%S)_$RANDOM"
fi

TRAIN_OPTS="federate.total_round_num ${TRAIN_ROUNDS}::train.local_update_steps ${LOCAL_STEPS}"

CMD=("$ROOT_DIR/scripts/run_local_individual_ablation_pipeline.sh"
  --run-tags "$RUN_TAG"
  --skip-eval
  --skip-collect
  --pipe-id "$PIPE_ID"
  --gpus "$GPUS"
  --results-root "$RESULTS_ROOT"
  --logs-root "$LOGS_ROOT"
  --train-opts "$TRAIN_OPTS"
)

if [[ -n "$CONDA_ENV" ]]; then
  CMD+=(--conda-env "$CONDA_ENV")
fi
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
  CMD+=(--hf-home "$HF_HOME_OVERRIDE")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=(--dry-run)
fi

printf "[%s] Launching sanity test: tag=%s rounds=%s local_steps=%s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "$RUN_TAG" "$TRAIN_ROUNDS" "$LOCAL_STEPS"
printf "[%s] Pipeline command: %q " "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${CMD[0]}"
for arg in "${CMD[@]:1}"; do
  printf "%q " "$arg"
done
printf "\n"

"${CMD[@]}"
