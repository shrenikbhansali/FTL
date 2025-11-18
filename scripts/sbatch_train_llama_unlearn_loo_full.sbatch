#!/bin/bash
#SBATCH --job-name=fsllm-train-llama-unlearn-loo-full
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

declare -a YAMLs=(
  "yamls/full/llama2_composite_meta3_unlearn_loo.yaml"
  "yamls/full/llama2_composite_meta3_unlearn_loo_shrink05.yaml"
)

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
export WANDB_API_KEY=78575c71e3200c01b61cae0272aef2d952ca18f9
export WANDB_ENTITY=sbhansali8-georgia-institute-of-technology
export WANDB_PROJECT=FTL-Dev
cd /home/hice1/sbhansali8/scratch/FederatedScope

CFG="${YAMLs[$SLURM_ARRAY_TASK_ID]}"
python federatedscope/main.py --cfg "$CFG"
