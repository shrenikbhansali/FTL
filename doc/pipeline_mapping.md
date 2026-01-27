# FTL pipeline script mapping + paper alignment

This note maps the FTL pipeline scripts to the experiment categories in the
FTL masterplan, and clarifies how (or whether) they correspond to the currently
open `DVI/docs/paper.tex`.

## Scope note: paper.tex vs FTL

- `DVI/docs/paper.tex` is a *different project* (Draft-Verify-Improve / DVI).
- It does **not** reference FTL experiments or pipelines.
- Therefore, there is **no direct correspondence** between the FTL scripts
  below and `paper.tex` as currently written.

If you intended an FTL paper, point to the correct `.tex` file and I can
link sections/tables directly.

## Masterplan alignment (FTL)

The FTL masterplan lives in:
- `FTL/doc/masterplan.md`

The scripts below implement the runs referenced in that plan.

## Primary pipeline scripts (FTL)

### Baselines (P0 main results)

- **Main baseline pipeline (global model)**
  - Script: `FTL/scripts/run_local_individual_pipeline.sh`
  - Runs: FedAvg, Bank-per-client, Centralized
  - Configs: `yamls/individual_federated_{fedavg,bank_perclient,centralized}.yaml`
  - Output: `FTL/individual_results/<PIPE_ID>/...`

- **Eval-only on an existing baseline run**
  - Script: `FTL/scripts/run_local_individual_baseline_eval_only.sh`
  - Use when training already done; re-evaluates checkpoints.

- **Single-task centralized ceilings**
  - Script: `FTL/scripts/run_local_individual_single_task_baselines.sh`
  - Uses centralized config with one client at a time.

- **Long-run variant**
  - Script: `FTL/scripts/run_local_individual_long_pipeline.sh`
  - Uses `individual_federated_long_*` configs.

### Specialization / retention (P0)

- **Specialization retention (per-client checkpoints)**
  - Script: `FTL/scripts/run_local_individual_specialization_pipeline.sh`
  - Evaluates per-client checkpoints on their own datasets.

- **All-pairs eval (per-client checkpoint vs all tasks)**
  - Script: `FTL/scripts/run_local_individual_allpairs_eval_pipeline.sh`

### Ablations (P1)

- **Full ablation suite**
  - Script: `FTL/scripts/run_local_individual_ablation_pipeline.sh`
  - Config root: `FTL/final/configs/ablations/`
  - Includes: decomposition-only, reweight-only, server-only, LoRA A/B/AB,
    projection mode/rank/rho, beta sweep.

- **Ablation sanity (short run)**
  - Script: `FTL/scripts/run_local_individual_ablation_sanity.sh`

- **Projection sanity (on/off)**
  - Script: `FTL/scripts/run_projection_sanity_pipeline.sh`

- **Specialization eval for projection ablation pair**
  - Script: `FTL/scripts/run_ablation_projection_specialization_eval.sh`
  - H200 variant: `FTL/scripts/run_ablation_projection_specialization_eval_h200.sh`

### Multi-client sharding / sweeps (P2)

- **Sharded pipeline**
  - Script: `FTL/scripts/run_local_individual_sharded_pipeline.sh`

- **Setting2 pipeline**
  - Script: `FTL/scripts/run_local_individual_setting2_pipeline.sh`

- **Multi-config sweeps**
  - Script: `FTL/scripts/run_local_individual_sweep_pipeline.sh`
  - Script: `FTL/scripts/run_local_individual_5multiconfig_pipeline.sh`
  - Script: `FTL/scripts/run_5multiconfig_eval_fanout.sh`

- **Task-aware sweeps (2-run / 5-run)**
  - Train: `FTL/scripts/run_local_individual_taskaware_2run_pipeline.sh`
  - Eval: `FTL/scripts/run_local_individual_taskaware_2run_eval_a40.sh`
  - Train: `FTL/scripts/run_local_individual_taskaware_5run_pipeline.sh`
  - Eval: `FTL/scripts/run_local_individual_taskaware_5run_eval_a40.sh`

- **Combo sweep**
  - Script: `FTL/scripts/run_local_individual_combo_pipeline.sh`

- **3-run sweep**
  - Train: `FTL/scripts/run_local_individual_3run_train_sweep.sh`
  - Eval: `FTL/scripts/run_local_individual_3run_eval_a40.sh`

## What to run (quick cookbook)

Use these commands from the FTL repo root (or prefix with `FTL/` if you are
elsewhere). You can override GPUs, IDs, HF cache, etc. via flags shown in
`--help` for each script.

### 1) Baseline global run (FedAvg, Bank-per-client, Centralized)

```bash
bash FTL/scripts/run_local_individual_pipeline.sh \
  --gpus 0,1,2,3 \
  --eval-max-samples 200
```

### 2) Single-task centralized ceilings

```bash
bash FTL/scripts/run_local_individual_single_task_baselines.sh \
  --gpus 0,1,2,3 \
  --eval-max-samples 200
```

### 3) Specialization retention (per-client ckpts)

```bash
bash FTL/scripts/run_local_individual_specialization_pipeline.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --eval-max-samples 200
```

### 4) Ablations (full suite)

```bash
bash FTL/scripts/run_local_individual_ablation_pipeline.sh \
  --gpus 0,1,2,3 \
  --eval-max-samples 200
```

### 5) Projection sanity

```bash
bash FTL/scripts/run_projection_sanity_pipeline.sh \
  --gpus 0,1
```

### 6) Multi-client sharded sweep

```bash
bash FTL/scripts/run_local_individual_sharded_pipeline.sh \
  --gpus 0,1,2,3 \
  --eval-max-samples 200
```

## Where outputs go

- Baseline/specialization (default): `FTL/individual_results/<PIPE_ID>/...`
- Final organized outputs (as per masterplan):
  - `FTL/final/results/`
  - `FTL/final/logs/`
  - `FTL/final/configs/`

## If you want a paper mapping

Tell me which FTL paper `.tex` file should be used, and I can map
sections/tables to the exact scripts and results paths.
