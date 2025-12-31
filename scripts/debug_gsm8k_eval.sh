#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE'
Usage: debug_gsm8k_eval.sh CKPT_PATH FP16[True|False] [MAX_SAMPLES]

Examples:
  ./scripts/debug_gsm8k_eval.sh ckpts/full/tulu3_federated_fedavg.ckpt False 32
  ./scripts/debug_gsm8k_eval.sh ckpts/full/tulu3_federated_fedavg.ckpt True 32
USAGE
  exit 1
fi

CKPT=$1
FP16=$2
MAX_SAMPLES=${3:-32}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RES_DIR="$ROOT_DIR/results_debug/gsm8k/$(date +%Y%m%d_%H%M%S)_fp16_${FP16}"
mkdir -p "$RES_DIR"

EVAL_YAML="$RES_DIR/eval_gsm8k_debug.yaml"
cat > "$EVAL_YAML" <<EOF
use_gpu: True
device: 0
outdir: "$RES_DIR/exp"
federate:
  save_to: "$CKPT"
model:
  type: "meta-llama/Llama-2-7b-hf@huggingface_llm"
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
  is_enable_half: $FP16
eval:
  max_samples: $MAX_SAMPLES
EOF

python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg "$EVAL_YAML"
