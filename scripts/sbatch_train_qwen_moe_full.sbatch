#!/bin/bash
#SBATCH --job-name=fsllm-train-qwen-full
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

python federatedscope/main.py --cfg yamls/full/qwen_moe_composite_meta3_full.yaml
