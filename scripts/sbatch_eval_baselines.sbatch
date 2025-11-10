#!/bin/bash
#SBATCH --job-name=fsllm-eval-baselines
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

# 8 baselines × 3 tasks
MODELS=(llama2 llama2 llama2 llama2 qwen_moe qwen_moe qwen_moe qwen_moe)
VARIANTS=(code_local gsm8k_local instr_local all_in_one code_local gsm8k_local instr_local all_in_one)
TASKS=(mmlu gsm8k code)

mid=$(( SLURM_ARRAY_TASK_ID / 3 ))
tid=$(( SLURM_ARRAY_TASK_ID % 3 ))
MODEL=${MODELS[$mid]}
VAR=${VARIANTS[$mid]}
TASK=${TASKS[$tid]}

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RES_DIR=$ROOT/results/baselines/$MODEL/$VAR/$TASK/$SLURM_JOB_ID
mkdir -p "$RES_DIR"

if [[ "$TASK" == "code" ]]; then
  CSN_FILE="$ROOT/data/csn_test.jsonl"
  LOCK_FILE="$ROOT/data/.csn.lock"
  if [[ ! -f "$CSN_FILE" ]]; then
    mkdir -p "$ROOT/data"
    (
      flock 200
      if [[ ! -f "$CSN_FILE" ]]; then
        python federatedscope/llm/dataset/code_search_net.py
      fi
    ) 200>"$LOCK_FILE"
  fi
fi

if [[ "$MODEL" == "llama2" ]]; then
  HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'
  CKPT="ckpts/baselines/${MODEL}_${VAR}.ckpt"
  FP16=True
else
  HF_TYPE='Qwen/Qwen1.5-MoE-A2.7B@huggingface_llm'
  CKPT="ckpts/baselines/${MODEL}_${VAR}.ckpt"
  FP16=False
fi

EVAL_YAML="$RES_DIR/eval_${MODEL}_${VAR}_${TASK}.yaml"
cat > "$EVAL_YAML" <<EOF
use_gpu: True
device: 0
outdir: "$RES_DIR/exp"
federate:
  save_to: "$CKPT"
model:
  type: "$HF_TYPE"
llm:
  tok_len: 1024
  adapter:
    use: True
    args:
      - {adapter_package: "peft", adapter_method: "lora",
         r: 8, lora_alpha: 32, lora_dropout: 0.05,
         target_modules: ["q_proj","k_proj","v_proj","o_proj"],
         modules_to_save: ["embed_tokens","lm_head"]}
train:
  is_enable_half: $FP16
eval:
  max_samples: 200
EOF

if [[ "$TASK" == "mmlu" ]]; then
  python federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg "$EVAL_YAML"
elif [[ "$TASK" == "gsm8k" ]]; then
  python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$EVAL_YAML"
else
  python federatedscope/llm/eval/eval_for_code/eval.py --cfg "$EVAL_YAML"
fi
