#!/bin/bash
#SBATCH --job-name=fsllm-prepare-superni-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

python scripts/prepare_superni_federated.py \
  --dataset Muennighoff/natural-instructions \
  --output-dir data/superni_federated \
  --group-by task_name \
  --dedupe-by id \
  --overwrite \
  --max-length 2048 \
  --max-total-samples 200000 \
  --max-samples-per-client 20000 \
  --val-fraction 0.01 \
  --seed 42 \
  --streaming
