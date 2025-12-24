#!/bin/bash
#SBATCH --job-name=fsllm-eval-tulu-global
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-15
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

EXPS=(fedavg bank_perclient centralized llama2_base)
TASKS=(mmlu gsm8k ifeval humaneval)

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
EXP=${EXPS[$EXP_IDX]}
TASK=${TASKS[$TASK_IDX]}

module load anaconda3
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RES_DIR=$ROOT/results_tulu/global/${EXP}/${TASK}/$SLURM_JOB_ID
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

case "$EXP" in
  fedavg)
    CKPT="ckpts/full/tulu3_federated_fedavg.ckpt"
    ADAPTER_USE=True
    ;;
  bank_perclient)
    CKPT="ckpts/full/tulu3_federated_unlearn_bank_perclient.ckpt"
    ADAPTER_USE=True
    ;;
  centralized)
    CKPT="ckpts/full/tulu3_federated_centralized.ckpt"
    ADAPTER_USE=True
    ;;
  *)
    CKPT=""
    ADAPTER_USE=False
    ;;
esac

HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'
FP16=True

EVAL_YAML="$RES_DIR/eval_${EXP}_${TASK}.yaml"
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
    use: $ADAPTER_USE
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
elif [[ "$TASK" == "ifeval" ]]; then
  python federatedscope/llm/eval/eval_for_ifeval/eval.py --cfg "$EVAL_YAML"
else
  python federatedscope/llm/eval/eval_for_humaneval/eval.py --cfg "$EVAL_YAML"
fi
