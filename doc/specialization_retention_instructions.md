# Specialization Retention + Single‑Task Ceiling Instructions

This document contains handoff instructions for generating early specialization
retention results and single‑task centralized ceilings. Scope is strictly the
five individual datasets: gsm8k, hellaswag, xsum, hotpotqa, mbpp.

## Background (where to look)

- Baseline configs:
  - `FTL/yamls/individual_federated_fedavg.yaml`
  - `FTL/yamls/individual_federated_bank_perclient.yaml`
  - `FTL/yamls/individual_federated_centralized.yaml`
- Client definitions:
  - `FTL/materials/individual_federated_clients.yaml`
  - `FTL/data/individual_federated/manifest.json`
- Baseline run results:
  - `FTL/individual_results/individual_local_20260115_100457_21987/`
- Eval entrypoints:
  - `FTL/federatedscope/llm/eval/eval_for_gsm8k/eval.py`
  - `FTL/federatedscope/llm/eval/eval_for_hellaswag/eval.py`
  - `FTL/federatedscope/llm/eval/eval_for_xsum/eval.py`
  - `FTL/federatedscope/llm/eval/eval_for_hotpotqa/eval.py`
  - `FTL/federatedscope/llm/eval/eval_for_mbpp/eval.py`
- Aggregation:
  - `FTL/scripts/collect_tulu_eval_results.py`
- Pipeline reference:
  - `FTL/scripts/run_local_individual_pipeline.sh`

## Task A — Specialization retention (FedAvg + Bank‑per‑client)

Goal: evaluate **per‑client checkpoints** on the **same client’s benchmark**.
These are the “specialization retention” numbers.

### What to do

1) Locate per‑client checkpoints for the **baseline** run:
   - Base dir: `FTL/individual_results/individual_local_20260115_100457_21987/`
   - Search under `fedavg/` and `bank_perclient/` for per‑client checkpoint
     files (often in `train/` subdirs).
2) For each dataset client (gsm8k, hellaswag, xsum, hotpotqa, mbpp):
   - Run the matching eval script on that client’s own checkpoint.
   - Use a small sample size for early results (e.g., 50–100).
3) Save the eval outputs in a consistent subfolder so they can be aggregated
   with `collect_tulu_eval_results.py` or a small custom summary script.

### Expected output

- A compact table with per‑client performance for:
  - **FedAvg** per‑client checkpoints
  - **Bank‑per‑client** per‑client checkpoints
- One value per benchmark (gsm8k accuracy, hellaswag accuracy, xsum ROUGE‑L,
  hotpotqa EM/F1, mbpp pass@1).

## Task B — Single‑task centralized ceilings (per‑task models)

Goal: train a single centralized model on **one dataset only** and evaluate
on **that dataset only**. This is the “ceiling” for specialization.

### What to do

1) Use `FTL/yamls/individual_federated_centralized.yaml` with
   `data.tulu3_federated.clients=[<one client name>]` for each dataset:
   - `gsm8k_client`
   - `hellaswag_client`
   - `xsum_client`
   - `hotpotqa_client`
   - `mbpp_client`
2) Train each dataset‑specific centralized run.
3) Evaluate only the matching benchmark for that run.

### Expected output

One table with “centralized single‑task” results per dataset/benchmark.

## Notes / tips

- Use the same model and LoRA settings as the baseline configs.
- Keep the eval scripts and output naming consistent with the existing
  pipeline so the notebook can ingest the results.
- Do not include Tulu, SuperNI, or P3 runs.
