# 5-Config Sharded Sweep Analysis

This note summarizes the 5-config sharded sweep under
`FTL/5multiconfig/results/5multiconfig_20260122_001523_21291` and the
corresponding logs in `FTL/5multiconfig/logs/5multiconfig_20260122_001523_21291`.
It connects hyperparameters, training behavior (loss + energy fractions),
and evaluation outcomes to guide the next iteration of sharded experiments.

## 1) Run inventory and data quality

**Run tags** (sharded setting):
- `participation_high`
- `participation_low`
- `more_shards`
- `drift_heavy`
- `balanced_shards`

**Primary tasks used in analysis** (valid metrics):
- `gsm8k` (weighted_accuracy)
- `hellaswag` (weighted_accuracy)
- `xsum` (rougeL_f1)
- `hotpotqa` (f1)
- `mbpp` (pass@1)

**Tasks excluded or flagged**:
- `piqa`: all runs report HF dataset script error (`piqa.py`).
- `apps`: all runs report HF dataset script error (`apps.py`).
- `toolbench`: env vars missing (no toolbench evaluator configured).

**Eval status** (most recent logs):
- Most evals finished; remaining failures are OOMs on A40 fanout runs:
  - `balanced_shards`: `fedavg/hellaswag` OOM
  - `drift_heavy`: `fedavg/mbpp` OOM
  - `more_shards`: `centralized/piqa` OOM (piqa invalid anyway)
- Some tasks have no log at all yet (not run), especially in
  `balanced_shards`, `drift_heavy`, and `more_shards` for the invalid tasks
  (piqa/apps/toolbench) and some gsm8k entries.

Implication: analysis below focuses on tasks with valid metrics and logs.

## 2) Hyperparameters (from train config.yaml)

| run | rounds | local_steps | sample_rate | sample_num | client_num | shards | sharding_spec | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_shards | 80 | 200 | 0.33 | 9 | 30 | 6 | gsm8k_client=6,hellaswag_client=8,xsum_client=8,hotpotqa_client=6,mbpp_client=2 | 0.0002 |
| drift_heavy | 50 | 320 | 0.35 | 10 | 30 | 6 |  | 0.0002 |
| more_shards | 80 | 200 | 0.25 | 10 | 40 | 8 |  | 0.0002 |
| participation_high | 80 | 200 | 0.5 | 15 | 30 | 6 |  | 0.0002 |
| participation_low | 80 | 200 | 0.25 | 7 | 30 | 6 |  | 0.0002 |

## 3) Bank energy fractions (bank_perclient, wandb-summary)

| run | global_fraction | private_fraction | resid_fraction |
| --- | --- | --- | --- |
| balanced_shards | 0.2205 | 0.6684 | 5.16e-12 |
| drift_heavy | 0.2447 | 0.6553 | 1.80e-11 |
| more_shards | 0.2704 | 0.6296 | 1.76e-11 |
| participation_high | 0.2208 | 0.7125 | 2.09e-11 |
| participation_low | 0.2122 | 0.6450 | 3.32e-12 |

Observations:
- Residual is near zero in all runs; bank updates are effectively split
  between global + private subspaces only.
- `participation_high` yields the highest private fraction, and the lowest
  (tied) global fraction.
- `more_shards` yields the highest global fraction and the lowest private
  fraction among the runs.

## 4) Training loss behavior (per-round mean train_avg_loss)

Computed from `exp_print.log` by averaging client train_avg_loss per round.

| run | exp | rounds | loss_first | loss_last | loss_mean | nan_batches | skipped_batches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_shards | fedavg | 80 | 1.2683 | 1.2350 | 1.0202 | 364 | 364 |
| balanced_shards | bank_perclient | 80 | 1.2683 | 1.2384 | 1.0270 | 364 | 364 |
| drift_heavy | fedavg | 50 | 0.9888 | 0.5955 | 0.8842 | 334 | 334 |
| drift_heavy | bank_perclient | 50 | 0.9888 | 0.5944 | 0.8871 | 334 | 334 |
| more_shards | fedavg | 80 | 0.8925 | 0.9688 | 0.8980 | 355 | 355 |
| more_shards | bank_perclient | 80 | 0.8925 | 0.9780 | 0.9011 | 355 | 355 |
| participation_high | fedavg | 80 | 0.9548 | 0.8028 | 0.8784 | 489 | 489 |
| participation_high | bank_perclient | 80 | 0.9548 | 0.8036 | 0.8859 | 489 | 489 |
| participation_low | fedavg | 80 | 1.0865 | 0.8177 | 0.9323 | 236 | 236 |
| participation_low | bank_perclient | 80 | 1.0865 | 0.8240 | 0.9336 | 236 | 236 |

Observations:
- FedAvg and bank_perclient have nearly identical loss trajectories within
  each run; loss alone does not explain differences in eval behavior.
- `drift_heavy` has the largest loss drop (local_steps=320) but no strong
  bank_perclient advantage on evals.
- NaN/skipped counts correlate with participation: `participation_high`
  shows the most NaNs (489), `participation_low` the fewest (236).

## 5) Eval outcomes for core tasks (valid metrics)

All values are from the latest `accuracies__*__<task>.json` file per task.

### participation_high

| task | centralized | fedavg | bank_perclient | bank - fed | bank - central |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 0.1387 | 0.0963 | 0.0910 | -0.0053 | -0.0478 |
| hellaswag | 0.5100 | 0.2200 | 0.1500 | -0.0700 | -0.3600 |
| xsum | 0.1662 | 0.1699 | 0.1701 | 0.0002 | 0.0039 |
| hotpotqa | 0.0694 | 0.1027 | 0.1080 | 0.0053 | 0.0387 |
| mbpp | 0.0000 | 0.0100 | 0.0200 | 0.0100 | 0.0200 |

### participation_low

| task | centralized | fedavg | bank_perclient | bank - fed | bank - central |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 0.1387 | 0.1152 | 0.1175 | 0.0023 | -0.0212 |
| hellaswag | 0.5100 | 0.1800 | 0.4400 | 0.2600 | -0.0700 |
| xsum | 0.1662 | 0.1588 | 0.1801 | 0.0213 | 0.0139 |
| hotpotqa | 0.0694 | 0.1130 | 0.1019 | -0.0111 | 0.0325 |
| mbpp | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### more_shards

| task | centralized | fedavg | bank_perclient | bank - fed | bank - central |
| --- | --- | --- | --- | --- | --- |
| gsm8k | - | 0.0917 | 0.0902 | -0.0015 | - |
| hellaswag | 0.5200 | 0.2600 | 0.3300 | 0.0700 | -0.1900 |
| xsum | 0.1696 | 0.1654 | 0.1656 | 0.0002 | -0.0039 |
| hotpotqa | 0.0655 | 0.1009 | 0.0962 | -0.0047 | 0.0307 |
| mbpp | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### drift_heavy

| task | centralized | fedavg | bank_perclient | bank - fed | bank - central |
| --- | --- | --- | --- | --- | --- |
| gsm8k | - | - | - | - | - |
| hellaswag | 0.5000 | 0.2300 | 0.2300 | 0.0000 | -0.2700 |
| xsum | 0.1668 | 0.1676 | 0.1561 | -0.0114 | -0.0106 |
| hotpotqa | 0.0711 | 0.1080 | 0.1137 | 0.0057 | 0.0426 |
| mbpp | 0.0000 | - | 0.0100 | - | 0.0100 |

### balanced_shards

| task | centralized | fedavg | bank_perclient | bank - fed | bank - central |
| --- | --- | --- | --- | --- | --- |
| gsm8k | - | - | - | - | - |
| hellaswag | 0.5200 | - | 0.3200 | - | -0.2000 |
| xsum | 0.1696 | 0.1752 | 0.1840 | 0.0088 | 0.0145 |
| hotpotqa | 0.0655 | 0.1032 | 0.1117 | 0.0085 | 0.0462 |
| mbpp | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0200 |

Key takeaways from results:
- `participation_high` (sample_rate=0.5) makes FedAvg strong and bank_perclient
  **weaker** on gsm8k/hellaswag; bank only slightly wins on hotpotqa/xsum.
- `participation_low` yields the largest **bank_perclient advantage** on
  hellaswag (+0.26 vs fedavg) and xsum (+0.021 vs fedavg). It also yields a
  small gsm8k improvement vs fedavg.
- `more_shards` (40 clients, shards=8) gives bank_perclient a modest
  improvement on hellaswag, but no meaningful gains on other tasks.
- `drift_heavy` (50 rounds, 320 steps) does not improve bank_perclient
  relative to fedavg; xsum slightly degrades.
- `balanced_shards` (non-IID client counts per task) shows modest bank_perclient
  wins on xsum/hotpotqa/mbpp, despite limited coverage due to missing gsm8k and
  a fedavg/hellaswag OOM.

## 6) Theoretical grounding (bank_perclient behavior)

Bank_perclient builds a shared basis from participating clients each round,
and projects per-client updates to avoid stepping into other clients' private
subspaces. Let Δ_i be the client update, and let it decompose into
Δ_i = Δ_shared + Δ_private + Δ_resid. Empirically the residual term is nearly
zero; the split is almost entirely between shared and private.

Implications:
1) **Participation rate m affects basis quality and FedAvg strength.**
   - As m (clients per round) increases, the shared basis is more stable, but
     FedAvg becomes closer to centralized (variance of the average update
     shrinks). This makes FedAvg stronger and narrows the bank advantage.
   - As m decreases, client drift increases and FedAvg weakens, but the shared
     basis is estimated from fewer clients, which can destabilize the bank.

2) **Local steps control drift magnitude.**
   - More local steps increase ||Δ_i|| and heterogeneity between clients.
   - In theory this should help bank_perclient (since it can suppress cross-
     client interference), but if too few rounds are used (drift_heavy), the
     shared basis may not stabilize and performance gains vanish.

3) **Sharding reduces per-client data, raising variance in Δ_i.**
   - More shards (more_shards) increases client count and decreases per-client
     data size, which can raise gradient variance and hurt both FedAvg and bank.
   - The slight rise in global_fraction for more_shards suggests a relatively
     stronger shared component, but it does not reliably translate into better
     bank_perclient metrics.

## 7) Conclusions for the next sharded iteration

**What appears to help bank_perclient:**
- Lower participation (`participation_low`) produced the clearest bank
  improvements (hellaswag, xsum) without breaking training stability.
- Non-IID sharding (`balanced_shards`) yielded bank gains on xsum/hotpotqa/mbpp,
  suggesting heterogeneity helps when basis estimation is still feasible.

**What appears to hurt or not help:**
- High participation (`participation_high`) compresses heterogeneity and makes
  FedAvg too strong, shrinking bank advantage.
- Drift-heavy setup (fewer rounds + more local steps) did not yield a bank
  advantage despite larger loss reduction.
- Increasing shards (`more_shards`) did not produce a consistent bank lift;
  the global fraction rose but evaluation improvements were modest.

**Key contradictions / uncertainty:**
- Energy fractions do not map cleanly to performance. Runs with higher private
  fraction (participation_high) can still have weak bank performance. This
  suggests the fraction alone is insufficient; basis quality and per-client data
  size likely dominate.
- Loss curves are nearly identical for FedAvg and bank_perclient across runs,
  so the bank effect is not visible in training loss; it only appears in evals.

## 8) Recommended next steps

Given the above, the next sweep should focus on isolating participation rate
and sharding degree while keeping total budget and steps stable:

1) **Participation sweep around 0.2–0.35** with fixed shards=6.
   - We already saw 0.25 outperforming 0.5. A finer sweep (0.2, 0.3, 0.35)
     will help locate the optimal basis/heterogeneity balance.

2) **Controlled sharding sweep** with fixed participation (0.25 or 0.3),
   comparing shards=6 vs 8 vs 10, while keeping total clients fixed.

3) **Non-IID spec ablation** (balanced_shards) with the same total client count,
   to quantify how task-skewed client distributions interact with the bank.

4) **Eval reliability fixes** (piqa/apps/toolbench):
   - Drop piqa/apps from analysis until HF dataset script issues are resolved.
   - Configure toolbench env or explicitly disable toolbench evals for clarity.

The goal for the next pass is to determine whether the bank advantage scales
with controlled heterogeneity or if it collapses once per-client data gets too
thin. The current data favors **moderate participation** plus **non-IID
client allocation**, but the evidence is not yet decisive.
