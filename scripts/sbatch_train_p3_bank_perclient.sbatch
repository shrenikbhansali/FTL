#!/bin/bash
#SBATCH --job-name=fsllm-train-p3-bank
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

EXP_NAME="bank_perclient"
RESULTS_ROOT="${P3_RESULTS_DIR:-$PWD/results_p3}"
OUTDIR="$RESULTS_ROOT/$EXP_NAME/train"
CKPT_PATH="$OUTDIR/ckpt.ckpt"
mkdir -p "$OUTDIR"

export TMPDIR="$OUTDIR/tmp"
export WANDB_DIR="$OUTDIR/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
unset WANDB_SERVICE
export WANDB_DISABLE_SERVICE=1
unset WANDB_SWEEP_ID WANDB_SWEEP_PARAM_PATH WANDB_CONFIG
export WANDB_USE=0
export WANDB_DISABLED=1
unset WANDB_RUN_ID

python federatedscope/main.py \
  --cfg yamls/p3_federated_bank_perclient.yaml \
  outdir "$OUTDIR" expname "$EXP_NAME" expname_tag "run_${P3_PIPE_ID:-}" \
  federate.save_to "$CKPT_PATH" \
  ${P3_TRAIN_OPTS//::/ }
