#!/usr/bin/env bash
set -euo pipefail

MODE="local"
PRECISION="bf16"
MAX_BATCHES=4
MAX_CLIENTS=2
LOAD_CKPT=0
DO_TRAIN_STEP=0
DO_GSM8K_GEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch)
      MODE="sbatch"
      shift
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --max-batches)
      MAX_BATCHES="$2"
      shift 2
      ;;
    --max-clients)
      MAX_CLIENTS="$2"
      shift 2
      ;;
    --load-ckpt)
      LOAD_CKPT=1
      shift
      ;;
    --train-step)
      DO_TRAIN_STEP=1
      shift
      ;;
    --gsm8k-gen)
      DO_GSM8K_GEN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: run_sanity_checks.sh [options]

Options:
  --sbatch         Submit as a single Slurm job.
  --precision      bf16 (default), fp16, or fp32.
  --max-batches    Batches per client to check (default: 4).
  --max-clients    Clients per config to check (default: 2).
  --load-ckpt      Load cfg.federate.save_to if it exists.
  --train-step     Run one backward/optimizer step per run.
  --gsm8k-gen      Run a tiny GSM8K generation sanity check.
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

OUT_DIR="$ROOT_DIR/results_debug/sanity"
mkdir -p "$OUT_DIR"

RUN_CMD=("$ROOT_DIR/scripts/sanity_check_training.py")
if [[ "$LOAD_CKPT" -eq 1 ]]; then
  RUN_CMD+=("--load-ckpt")
fi
if [[ "$DO_TRAIN_STEP" -eq 1 ]]; then
  RUN_CMD+=("--do-train-step")
fi
if [[ "$DO_GSM8K_GEN" -eq 1 ]]; then
  RUN_CMD+=("--do-gsm8k-gen")
fi

if [[ "$MODE" == "sbatch" ]]; then
  sbatch "$ROOT_DIR/scripts/sbatch_sanity_checks.sbatch" \
    "$PRECISION" "$MAX_BATCHES" "$MAX_CLIENTS" "$LOAD_CKPT" \
    "$DO_TRAIN_STEP" "$DO_GSM8K_GEN"
  exit 0
fi

timestamp=$(date +%Y%m%d_%H%M%S)

SUMMARY_OUT="$OUT_DIR/sanity_summary_${timestamp}.json"

python "${RUN_CMD[@]}" --cfg yamls/tulu3_federated_centralized.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/centralized_${timestamp}.json"

python "${RUN_CMD[@]}" --cfg yamls/tulu3_federated_fedavg.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/fedavg_${timestamp}.json"

python "${RUN_CMD[@]}" --cfg yamls/tulu3_federated_unlearn_bank_perclient.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/bank_${timestamp}.json"

python scripts/sanity_check_summary.py \
  --inputs \
    "$OUT_DIR/centralized_${timestamp}.json" \
    "$OUT_DIR/fedavg_${timestamp}.json" \
    "$OUT_DIR/bank_${timestamp}.json" \
  --out "$SUMMARY_OUT"

echo "Sanity reports saved in $OUT_DIR"
echo "Summary report: $SUMMARY_OUT"
