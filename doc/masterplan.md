# FTL Paper Masterplan (Individual Datasets Only)

This checklist covers all experiments, ablations, and evaluations still needed
for the paper. Scope is strictly the five individual datasets:
gsm8k, hotpotqa, hellaswag, xsum, mbpp. Do not use Tulu, SuperNI, or P3 runs.

## P0 — Must Run (Main Results + Tables)

- [ ] Re-run Single-Client baseline (global model results) with higher eval
      samples (>=100 per dataset; current baseline used 50). Use the same
      configs as the paper baseline:
      - `FTL/yamls/individual_federated_centralized.yaml`
      - `FTL/yamls/individual_federated_fedavg.yaml`
      - `FTL/yamls/individual_federated_bank_perclient.yaml`
      - Evaluate GSM8K, HellaSwag, XSum, HotPotQA, MBPP.
- [ ] Single-task centralized ceilings (one dataset per run):
      - Use `FTL/yamls/individual_federated_centralized.yaml` with
        `data.tulu3_federated.clients=[<one client name>]` for each dataset.
      - Evaluate only the matching benchmark for each run and fill the
        “centralized single-task” column.
- [ ] Specialization retention (Single-Client only):
      - Evaluate per-client checkpoints for FedAvg and SubspaceBank on each
        client’s own dataset.
      - Fill Table “Specialization Retention”.
- [ ] Multi-Client main run (global model):
      - Pick ONE sharded setting as the main result (see P1), align rounds with
        final paper, and run centralized/fedavg/bank for that setting.
      - Evaluate GSM8K, HellaSwag, XSum, HotPotQA, MBPP.

## P1 — Priority Ablations (Necessity Grid + Key Sensitivities)

- [ ] Necessity grid (Single-Client regime; global evals):
      - FedAvg baseline (done in P0).
      - Decomposition-only: set `beta_global=1`, `beta_resid=1`.
      - Reweight-only: set `r_client=0`.
      - SubspaceBank server-only: disable client projection
        (`train.unlearn.project_grads=False`).
      - SubspaceBank full: baseline `bank_perclient`.
- [ ] A-only vs B-only vs A+B banking:
      - Set `aggregator.unlearn.only_lora=False` and
        `target_modules=["lora_A"]`, `["lora_B"]`, and both.
- [ ] Projection ablations:
      - `proj_rho`: {0, 0.5, 1.0}
      - `bank_proj_mode`: {others_private, others_plus_shared}
      - `bank_proj_rank_max`: {0, 8, 16}
- [ ] Geometry + leakage diagnostics:
      - Enable `train.unlearn.log_leakage=True` and
        `aggregator.unlearn.bank.log_geometry=True`
      - Run at least the main SubspaceBank baseline + key ablations.

## P2 — Multi-Client Sensitivity + Participation

- [ ] Multi-Client sharding/participation sweeps:
      - `FTL/yamls/individual_federated_sharded_light_*`
      - `FTL/yamls/individual_federated_sharded_setting2_*`
      - `FTL/yamls/individual_federated_sharded_setting3_*`
      - Each setting should include centralized, fedavg, bank_perclient.
- [ ] Partial participation (Single-Client regime):
      - `FTL/yamls/individual_federated_participation_*`
      - Check whether performance tracks participation-rate assumptions.
- [ ] Budget-matched baselines:
      - `FTL/yamls/individual_federated_budget_*`
      - Include global evals and compare to baseline to normalize compute.

## P3 — Curves, Diagnostics, and Robustness

- [ ] Convergence-vs-round plots:
      - Save periodic checkpoints and re-evaluate at multiple rounds.
      - Plot global performance vs round for FedAvg vs SubspaceBank.
- [ ] Geometry correlation plots:
      - Correlate `global_fraction`, `private_fraction`, `resid_fraction`
        (and leakage) with performance deltas vs FedAvg.
- [ ] Orthogonality / cross-private summaries:
      - Use logged geometry stats to summarize stability vs participation.

## P4 — Plotting Ideas Beyond Tables

- [ ] Radar/spider plot for the 5 primary metrics (GSM8K, HellaSwag, XSum,
      HotPotQA, MBPP) comparing centralized, FedAvg, SubspaceBank.
- [ ] Bar chart of macro-average vs method with error bars (if you have
      multiple seeds).
- [ ] Scatter plots: performance gain vs `global_fraction` / `resid_fraction`.
- [ ] Line plots: convergence curves (main + appendix).

## Results Organization Plan (Final Packaging)

Create a final, clean output tree to avoid mixing with sweeps:

```
FTL/final/
  configs/
    single_client/
    multi_client/
    ablations/
  results/
    single_client/
      global/
      client_specialization/
    multi_client/
    ablations/
    diagnostics/
  plots/
    main/
    appendix/
  tables/
  logs/
  data/
    eval_outputs/
    manifests/
```

Guidelines
- Copy YAMLs used for each run into `FTL/final/configs/...` with a short tag.
- Store `eval_result/accuracies_*.json`, `eval/aggregate.json`, and any logs in
  `FTL/final/results/...`.
- Place generated plots into `FTL/final/plots/...` and table CSVs into
  `FTL/final/tables/`.
- Keep `FTL/final/data/eval_outputs/` for any cached eval data artifacts.

Notes
- All runs should be restricted to the five individual datasets (gsm8k,
  hellaswag, xsum, hotpotqa, mbpp). Avoid Tulu, SuperNI, P3 configs.
- If you need new configs for ablations, clone baseline YAMLs and save to
  `FTL/final/configs/ablations/` with explicit names.
