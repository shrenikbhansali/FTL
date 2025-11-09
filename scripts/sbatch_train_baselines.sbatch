#!/bin/bash
#SBATCH --job-name=fsllm-train-baselines
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

declare -a YAMLs=(
  "yamls/baselines/llama2_code_local.yaml"
  "yamls/baselines/llama2_gsm8k_local.yaml"
  "yamls/baselines/llama2_instr_local.yaml"
  "yamls/baselines/llama2_all_in_one.yaml"
  "yamls/baselines/qwen_moe_code_local.yaml"
  "yamls/baselines/qwen_moe_gsm8k_local.yaml"
  "yamls/baselines/qwen_moe_instr_local.yaml"
  "yamls/baselines/qwen_moe_all_in_one.yaml"
)

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

CFG="${YAMLs[$SLURM_ARRAY_TASK_ID]}"
python federatedscope/main.py --cfg "$CFG"
