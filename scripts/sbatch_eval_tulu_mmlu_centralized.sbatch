#!/bin/bash
#SBATCH --job-name=fsllm-eval-tulu-mmlu-centralized
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RES_DIR=$ROOT/results_tulu/global/centralized/mmlu/$SLURM_JOB_ID
mkdir -p "$RES_DIR"

CKPT="ckpts/full/tulu3_federated_centralized.ckpt"
HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'

EVAL_YAML="$RES_DIR/eval_centralized_mmlu.yaml"
cat > "$EVAL_YAML" <<EOF
use_gpu: True
device: 0
outdir: "$RES_DIR/exp"
federate:
  save_to: "$CKPT"
model:
  type: "$HF_TYPE"
llm:
  tok_len: 2048
  adapter:
    use: True
    args:
      - {adapter_package: "peft", adapter_method: "lora",
         r: 8, lora_alpha: 32, lora_dropout: 0.05,
         target_modules: ["q_proj","k_proj","v_proj","o_proj"],
         modules_to_save: ["embed_tokens","lm_head"]}
train:
  is_enable_half: False
  precision: bf16
  compile: False
eval:
  max_samples_per_subject: 2
EOF

python federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg "$EVAL_YAML"
