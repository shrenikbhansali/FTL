#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTBASE="$ROOT/overhead/${RUN_TAG}"
mkdir -p "$OUTBASE"/{fedavg,subspacebank_serveronly}

# You can override these with environment variables when invoking the script.
FEDAVG_GPU="${FEDAVG_GPU:-0}"
BANK_GPU="${BANK_GPU:-1}"

export WANDB_DISABLED=1
export WANDB_DISABLE_SERVICE=1
export WANDB_USE=0

EXTRA_OPTS=()
if [[ "${COUNT_FLOPS:-0}" -eq 1 ]]; then
  EXTRA_OPTS+=(eval.count_flops True)
fi

echo "Launching FedAvg on GPU ${FEDAVG_GPU}"
CUDA_VISIBLE_DEVICES="${FEDAVG_GPU}" \
  python federatedscope/main.py \
    --cfg yamls/individual_federated_fedavg.yaml \
    outdir "$OUTBASE/fedavg" \
    expname "overhead_fedavg" \
    federate.save_to "$OUTBASE/fedavg/ckpt.ckpt" \
    "${EXTRA_OPTS[@]}" \
    >"$OUTBASE/fedavg/train.log" 2>&1 &

echo "Launching SubspaceBank (server-only) on GPU ${BANK_GPU}"
CUDA_VISIBLE_DEVICES="${BANK_GPU}" \
  python federatedscope/main.py \
    --cfg yamls/individual_federated_bank_perclient.yaml \
    train.unlearn.project_grads False \
    aggregator.unlearn.send_Q_to_clients False \
    outdir "$OUTBASE/subspacebank_serveronly" \
    expname "overhead_subspacebank_serveronly" \
    federate.save_to "$OUTBASE/subspacebank_serveronly/ckpt.ckpt" \
    "${EXTRA_OPTS[@]}" \
    >"$OUTBASE/subspacebank_serveronly/train.log" 2>&1 &

wait
echo "Done. Logs:"
echo "  $OUTBASE/fedavg/train.log"
echo "  $OUTBASE/subspacebank_serveronly/train.log"
echo "Run tag: $RUN_TAG"
