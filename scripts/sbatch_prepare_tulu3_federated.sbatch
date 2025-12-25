#!/bin/bash
#SBATCH --job-name=fsllm-prepare-tulu-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
conda init bash >/dev/null 2>&1 || true
source ~/.bashrc
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
export WANDB_API_KEY=78575c71e3200c01b61cae0272aef2d952ca18f9
export WANDB_ENTITY=sbhansali8-georgia-institute-of-technology
export WANDB_PROJECT=FTL-Dev
cd /home/hice1/sbhansali8/scratch/FederatedScope

python scripts/prepare_tulu3_federated.py \
  --output-dir data/tulu3_federated \
  --enable-length-filter \
  --filter-tokenizer-name meta-llama/Llama-2-7b-hf \
  --filter-max-length 2048 \
  --filter-min-assistant-tokens 1 \
  --overwrite
