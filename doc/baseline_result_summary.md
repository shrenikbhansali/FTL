# Baseline Results Summary

This document summarizes the baseline individual‑dataset experiments, including
datasets, training hyperparameters, and benchmark results.

## Baseline run

- Run id: `individual_local_20260115_100457_21987`
- Results root: `FTL/individual_results/individual_local_20260115_100457_21987`
- Eval job id: `eval_individual_local_20260115_100457_21987`
- Eval samples: 100 per benchmark (where available)

## Datasets (one client per task)

Configured in `FTL/materials/individual_federated_clients.yaml` and prepared in
`data/individual_federated`:

| Client | Dataset | Split | Style | Formatter |
| --- | --- | --- | --- | --- |
| gsm8k_client | gsm8k (main) | train | qa | gsm8k |
| hellaswag_client | hellaswag | train | mcq | hellaswag |
| xsum_client | xsum | train | summarization | xsum |
| mbpp_client | mbpp | train (+ validation) | code | mbpp |
| hotpotqa_client | hotpot_qa (distractor) | train | qa | hotpotqa |

## Training setup

### Model

- Base model: `meta-llama/Llama-2-7b-hf`
- LoRA: r=8, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Tok length: 2048

### Federated baselines

Configs:
- FedAvg: `FTL/yamls/individual_federated_fedavg.yaml`
- Bank‑per‑client: `FTL/yamls/individual_federated_bank_perclient.yaml`
- Centralized: `FTL/yamls/individual_federated_centralized.yaml`

Key hyperparameters:

| Setting | FedAvg | Bank‑per‑client | Centralized |
| --- | --- | --- | --- |
| total_round_num | 60 | 60 | 60 |
| local_update_steps | 250 | 250 | 75 |
| batch_size | 1 | 1 | 1 |
| precision | bf16 | bf16 | bf16 |
| lr | 2e‑4 | 2e‑4 | 2e‑5 |
| sample_client_rate | 1.0 | 1.0 | n/a (merge_clients) |

## Benchmark results (baseline)

Values are taken from:
- `FTL/individual_results/individual_local_20260115_100457_21987/centralized/eval/aggregate.json`
- `FTL/individual_results/individual_local_20260115_100457_21987/fedavg/eval/aggregate.json`
- `FTL/individual_results/individual_local_20260115_100457_21987/bank_perclient/eval/aggregate.json`

### Metrics table

| Benchmark | Metric | Centralized | FedAvg | Bank‑per‑client |
| --- | --- | --- | --- | --- |
| gsm8k | accuracy | 0.1357 | 0.0713 | 0.1251 |
| hellaswag | accuracy | 0.4600 | 0.1500 | 0.3600 |
| xsum | rougeL_f1 | 0.1791 | 0.1635 | 0.1832 |
| hotpotqa | EM | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | F1 | 0.0790 | 0.1237 | 0.1304 |
| mbpp | pass@1 | 0.0000 | 0.0100 | 0.0000 |

### Notes on missing or failed benchmarks

- piqa: failed to load dataset (dataset script not supported).
- apps: failed to load dataset (dataset script not supported).
- toolbench: evaluation not run (missing `TOOLBENCH_EVAL_CMD/TOOLBENCH_EVAL_OUTPUT`).

These failures are reported in each `aggregate.json` and should be excluded
from baseline comparisons.

## Summary observations

- FedAvg underperforms centralized across most benchmarks.
- Bank‑per‑client improves over FedAvg on gsm8k, hellaswag, xsum, and hotpotqa F1.
- Centralized remains strongest overall, but bank‑per‑client closes part of the gap
  under client heterogeneity.
