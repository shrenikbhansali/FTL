#!/bin/bash
#SBATCH --job-name=fsllm-eval-tulu-worstcase
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=224G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sbhansali8@gatech.edu

EXPS=(fedavg bank_perclient centralized)
TASKS=(mmlu gsm8k humaneval)
MMLU_MAX_SAMPLES_PER_SUBJECT=2
DEFAULT_MAX_SAMPLES=200

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
EXP=${EXPS[$EXP_IDX]}
TASK=${TASKS[$TASK_IDX]}
if [[ "$TASK" == "mmlu" ]]; then
  MAX_SAMPLES=$DEFAULT_MAX_SAMPLES
else
  MAX_SAMPLES=$DEFAULT_MAX_SAMPLES
fi

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fs-llm
export HF_HOME=/home/hice1/sbhansali8/scratch
cd /home/hice1/sbhansali8/scratch/FederatedScope

ROOT=$(pwd)
RESULTS_ROOT="${TULU_RESULTS_DIR:-$ROOT/results_tulu}"
RES_DIR=$RESULTS_ROOT/global/${EXP}/${TASK}/$SLURM_JOB_ID
mkdir -p "$RES_DIR"
TMP_BASE="$ROOT/tulupipe_tmp"
mkdir -p "$TMP_BASE"
export TMPDIR="$TMP_BASE/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export WANDB_DIR="$RES_DIR/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
unset WANDB_SERVICE

case "$EXP" in
  fedavg|bank_perclient|centralized)
    CKPT="$RESULTS_ROOT/$EXP/train/ckpt.ckpt"
    ADAPTER_USE=True
    ;;
  *)
    echo "Unknown experiment: $EXP" >&2
    exit 1
    ;;
esac

FINAL_CKPT="$RESULTS_ROOT/$EXP/train/final_ckpt.ckpt"
if [[ -f "$FINAL_CKPT" ]]; then
  CKPT="$FINAL_CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

HF_TYPE='meta-llama/Llama-2-7b-hf@huggingface_llm'
PRECISION='bf16'

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
  is_enable_half: False
  precision: $PRECISION
  compile: False
eval:
  max_samples: $MAX_SAMPLES
  max_samples_per_subject: $MMLU_MAX_SAMPLES_PER_SUBJECT
EOF

if [[ "$TASK" == "mmlu" ]]; then
  python federatedscope/llm/eval/eval_for_mmlu/eval.py --cfg "$EVAL_YAML"
elif [[ "$TASK" == "gsm8k" ]]; then
  python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$EVAL_YAML"
else
  python federatedscope/llm/eval/eval_for_humaneval/eval.py --cfg "$EVAL_YAML"
fi
