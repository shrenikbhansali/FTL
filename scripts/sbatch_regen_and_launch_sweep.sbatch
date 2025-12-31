#!/bin/bash
#SBATCH --job-name=tulu-regen-sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=tulupipe_logs/regen_sweep_%A.out
#SBATCH --error=tulupipe_logs/regen_sweep_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm

export HF_HOME=/home/hice1/sbhansali8/scratch
ROOT_DIR=/home/hice1/sbhansali8/scratch/FederatedScope
cd "$ROOT_DIR"
mkdir -p tulupipe_logs

echo "[regen] Generating group-level dataset..."
python scripts/prepare_tulu3_federated.py \
  --output-dir data/tulu3_federated_group \
  --group-by family \
  --client-config materials/tulu3_federated_clients.yaml \
  --enable-length-filter \
  --filter-tokenizer-name meta-llama/Llama-2-7b-hf \
  --filter-max-length 2048 \
  --filter-min-assistant-tokens 1 \
  --overwrite

echo "[regen] Generating source-level dataset..."
python scripts/prepare_tulu3_federated.py \
  --output-dir data/tulu3_federated_source \
  --group-by source \
  --enable-length-filter \
  --filter-tokenizer-name meta-llama/Llama-2-7b-hf \
  --filter-max-length 2048 \
  --filter-min-assistant-tokens 1 \
  --overwrite

echo "[regen] Submitting W&B agent jobs..."
H200_JOB=$(sbatch --parsable --export=ALL,TULU_GPU_TYPE=H200 scripts/sbatch_wandb_agent.sbatch)
H100_JOB=$(sbatch --parsable --export=ALL,TULU_GPU_TYPE=H100 scripts/sbatch_wandb_agent.sbatch)
echo "[regen] Submitted agent jobs: H200=${H200_JOB} H100=${H100_JOB}"
