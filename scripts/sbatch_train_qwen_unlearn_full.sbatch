#!/bin/bash
#SBATCH --job-name=fsllm-train-qwen-unlearn-full
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

declare -a YAMLs=(
  "yamls/full/qwen_moe_composite_meta3_unlearn.yaml"
  "yamls/full/qwen_moe_composite_meta3_unlearn_shrink05.yaml"
)

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

CFG="${YAMLs[$SLURM_ARRAY_TASK_ID]}"
python federatedscope/main.py --cfg "$CFG"
