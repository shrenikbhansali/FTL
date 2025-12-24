#!/bin/bash
#SBATCH --job-name=fsllm-train-tulu-centralized
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
export WANDB_API_KEY=78575c71e3200c01b61cae0272aef2d952ca18f9
export WANDB_ENTITY=sbhansali8-georgia-institute-of-technology
export WANDB_PROJECT=FTL-Dev
cd /home/hice1/sbhansali8/scratch/FederatedScope

python federatedscope/main.py --cfg yamls/tulu3_federated_centralized.yaml
