#!/bin/bash
#SBATCH --job-name=fsllm-prepare-p3-data
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

python scripts/prepare_p3_federated.py \
  --output-dir "${P3_DATA_ROOT:-data/p3_federated}" \
  --group-by "${P3_GROUP_BY:-dataset}" \
  ${P3_CONFIG_FILE:+--config-file "$P3_CONFIG_FILE"} \
  ${P3_CONFIGS:+--configs "$P3_CONFIGS"} \
  ${P3_CATEGORY_MAP:+--category-map "$P3_CATEGORY_MAP"} \
  --overwrite \
  --max-length 2048 \
  --max-total-samples 200000 \
  --max-samples-per-client 20000 \
  ${P3_MAX_CLIENTS:+--max-clients "$P3_MAX_CLIENTS"} \
  --val-fraction 0.01 \
  --seed 42 \
  --streaming
