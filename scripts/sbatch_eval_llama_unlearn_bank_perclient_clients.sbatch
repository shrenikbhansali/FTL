#!/bin/bash
#SBATCH --job-name=fsllm-eval-llama-bank-clients
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

TAG="unlearn_bank_perclient"
BASE_PATH="ckpts/full/llama2_composite_meta3_${TAG}"
HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'
FP16=False
CLIENTS=(1 2 3)
TASKS=("mmlu" "gsm8k" "ifeval" "humaneval")

TASKS_PER_CLIENT=${#TASKS[@]}
CLIENT_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_CLIENT))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % TASKS_PER_CLIENT))

CID=${CLIENTS[$CLIENT_IDX]}
TASK=${TASKS[$TASK_IDX]}
CKPT="${BASE_PATH}/clients/client_${CID}/client_${CID}.ckpt"

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RES_DIR=$ROOT/results_full/clients/llama2_${TAG}/client_${CID}/${TASK}/$SLURM_JOB_ID
mkdir -p "$RES_DIR"

if [[ "$TASK" == "ifeval" ]]; then
  IFEVAL_FILE="$ROOT/data/ifeval_input_data.jsonl"
  LOCK_FILE="$ROOT/data/.ifeval.lock"
  if [[ ! -f "$IFEVAL_FILE" ]]; then
    mkdir -p "$ROOT/data"
    (
      flock 200
      if [[ ! -f "$IFEVAL_FILE" ]]; then
        python federatedscope/llm/dataset/ifeval.py
      fi
    ) 200>"$LOCK_FILE"
  fi
fi

EVAL_YAML="$RES_DIR/eval_c${CID}_${TASK}.yaml"
cat > "$EVAL_YAML" <<EOF2
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
EOF2

if [[ "$TASK" == "mmlu" ]]; then
  python federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg "$EVAL_YAML"
elif [[ "$TASK" == "gsm8k" ]]; then
  python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$EVAL_YAML"
elif [[ "$TASK" == "ifeval" ]]; then
  python federatedscope/llm/eval/eval_for_ifeval/eval.py --cfg "$EVAL_YAML"
else
  python federatedscope/llm/eval/eval_for_humaneval/eval.py --cfg "$EVAL_YAML"
fi
