#!/bin/bash
#SBATCH --job-name=fsllm-eval-unlearn-clients-full
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-47
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

TASKS=("mmlu" "gsm8k" "ifeval" "humaneval")
CLIENTS=(1 2 3)
COMBO_MODELS=("llama2" "llama2" "qwen15moe" "qwen15moe")
COMBO_TAGS=("unlearn" "unlearn_shrink05" "unlearn" "unlearn_shrink05")

TASKS_PER_CLIENT=${#TASKS[@]}
CLIENTS_PER_COMBO=${#CLIENTS[@]}
PER_COMBO=$((TASKS_PER_CLIENT * CLIENTS_PER_COMBO))

COMBO_IDX=$((SLURM_ARRAY_TASK_ID / PER_COMBO))
WITHIN_COMBO=$((SLURM_ARRAY_TASK_ID % PER_COMBO))
CLIENT_IDX=$((WITHIN_COMBO / TASKS_PER_CLIENT))
TASK_IDX=$((WITHIN_COMBO % TASKS_PER_CLIENT))

MODEL=${COMBO_MODELS[$COMBO_IDX]}
TAG=${COMBO_TAGS[$COMBO_IDX]}
CID=${CLIENTS[$CLIENT_IDX]}
TASK=${TASKS[$TASK_IDX]}

if [[ "$MODEL" == "llama2" ]]; then
  BASE_PATH="ckpts/full/llama2_composite_meta3_${TAG}"
  HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'
  FP16=False
else
  BASE_PATH="ckpts/full/qwen15moe_composite_meta3_${TAG}"
  HF_TYPE='Qwen/Qwen1.5-MoE-A2.7B@huggingface_llm'
  FP16=False
fi
CKPT="${BASE_PATH}/clients/client_${CID}/client_${CID}.ckpt"

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RES_DIR=$ROOT/results_full/clients/${MODEL}_${TAG}/client_${CID}/${TASK}/$SLURM_JOB_ID
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

EVAL_YAML="$RES_DIR/eval_${MODEL}_${TAG}_c${CID}_${TASK}.yaml"
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
