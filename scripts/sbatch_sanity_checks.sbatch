#!/bin/bash
#SBATCH --job-name=fsllm-sanity-checks
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

PRECISION=${1:-bf16}
MAX_BATCHES=${2:-4}
MAX_CLIENTS=${3:-2}
LOAD_CKPT=${4:-0}
DO_TRAIN_STEP=${5:-0}
DO_GSM8K_GEN=${6:-0}

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

OUT_DIR="$(pwd)/results_debug/sanity"
mkdir -p "$OUT_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
SUMMARY_OUT="$OUT_DIR/sanity_summary_${timestamp}.json"

LOAD_FLAG=""
if [[ "$LOAD_CKPT" -eq 1 ]]; then
  LOAD_FLAG="--load-ckpt"
fi
TRAIN_FLAG=""
if [[ "$DO_TRAIN_STEP" -eq 1 ]]; then
  TRAIN_FLAG="--do-train-step"
fi
GSM8K_FLAG=""
if [[ "$DO_GSM8K_GEN" -eq 1 ]]; then
  GSM8K_FLAG="--do-gsm8k-gen"
fi

python scripts/sanity_check_training.py --cfg yamls/tulu3_federated_centralized.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/centralized_${timestamp}.json" \
  $LOAD_FLAG $TRAIN_FLAG $GSM8K_FLAG

python scripts/sanity_check_training.py --cfg yamls/tulu3_federated_fedavg.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/fedavg_${timestamp}.json" \
  $LOAD_FLAG $TRAIN_FLAG $GSM8K_FLAG

python scripts/sanity_check_training.py --cfg yamls/tulu3_federated_unlearn_bank_perclient.yaml \
  --precision "$PRECISION" \
  --max-batches "$MAX_BATCHES" \
  --max-clients "$MAX_CLIENTS" \
  --out "$OUT_DIR/bank_${timestamp}.json" \
  $LOAD_FLAG $TRAIN_FLAG $GSM8K_FLAG

python scripts/sanity_check_summary.py \
  --inputs \
    "$OUT_DIR/centralized_${timestamp}.json" \
    "$OUT_DIR/fedavg_${timestamp}.json" \
    "$OUT_DIR/bank_${timestamp}.json" \
  --out "$SUMMARY_OUT"

echo "Summary report: $SUMMARY_OUT"
