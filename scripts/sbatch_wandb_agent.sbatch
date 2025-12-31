#!/bin/bash
#SBATCH --job-name=wandb-agent
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=8:00:00
#SBATCH --output=tulupipe_logs/wandb_agent_%A_%a.out
#SBATCH --error=tulupipe_logs/wandb_agent_%A_%a.err

module load anaconda3

cd /home/hice1/sbhansali8/scratch/FederatedScope
mkdir -p tulupipe_logs

# Run one agent with a fixed number of runs
conda run -n fs-llm wandb agent --count 10 \
  sbhansali8-georgia-institute-of-technology/FederatedScope-scripts/2m1rksoa
