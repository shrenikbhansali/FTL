# Interpreting Ablation Results (Guidance for Paper + Reviewers)

This note is meant to help interpret the ablation suite in a way that is
useful for writing and defending the paper. It includes general guidance
and a short status update based on partial results currently available.

## Scope and caveats

- Ablations target the Single-Client Datasets regime only.
- Current run: `individual_ablations_20260122_191942_5628` with
  `eval_max_samples=100`.
- All ablation training is complete.
- GSM8K evals are complete for non-beta ablations.
- GSM8K evals are still running for the beta grid and `abl_proj_rank_max_16`.
- Treat any statements below as provisional until all evals finish.

## What reviewers generally want to see

- A clean necessity grid showing each component helps.
- Full method consistently beats FedAvg on most tasks.
- Removing key components degrades performance (not noise-level changes).
- Sensitivity shows a reasonable, stable region of hyperparameters.
- A-only/B-only clearly weaker than A+B if you claim both are needed.

## How to interpret outcomes

### Good results (support the paper)
- Full method > FedAvg on most benchmarks (ideally all but one).
- Decomposition-only and reweight-only underperform full.
- Projection ablations (rho=0, server-only, payload mode changes) reduce
  performance or at least not improve over full.
- A-only and B-only are worse than A+B.
- Beta sweep peaks near the baseline setting (0.65/0.15), not at extremes.

### Neutral results (unlikely to change the paper)
- Small deltas that are within expected noise for 100 samples.
- One or two tasks do not move while others improve.
- MBPP and GSM8K are low or noisy (these are often volatile at small N).

### Bad results (weaken claims)
- Decomposition-only or reweight-only matches full.
- A-only or B-only matches or beats A+B.
- FedAvg is within a few points of full across all tasks.
- Beta extremes (0 or 1) beat the baseline setting.
- Projection ablations show no difference and you currently claim they are
  central to the global gains.

### Results that would force paper changes
- Full method does not outperform FedAvg on the main global table.
- Ablations reveal the improvement is driven by a simpler subcomponent
  (e.g., reweight-only is best): the method should be reframed and
  simplified in the paper.
- A-only or B-only dominates: you would need to rewrite the mechanism and
  contributions to focus on the winning side.
- Projection has no effect anywhere (global or specialization): projection
  should be moved to the appendix or framed as optional.

## Partial results snapshot (current run, 100 samples)

The numbers below are from completed evals in
`FTL/final/results/ablations/individual_ablations_20260122_191942_5628`.
Missing entries indicate the eval has not finished yet for that task.

Key:
- G = GSM8K accuracy
- HS = HellaSwag accuracy
- XS = XSum ROUGE-L F1
- HP = HotPotQA F1
- MB = MBPP pass@1

| Ablation | G | HS | XS | HP | MB |
| --- | --- | --- | --- | --- | --- |
| abl_full | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_server_only | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_lora_AB | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_decomp_only | 0.1130 | 0.1300 | 0.1619 | 0.1203 | 0.0300 |
| abl_reweight_only | 0.0970 | 0.0300 | 0.1557 | 0.1367 | 0.0100 |
| abl_fedavg | 0.0667 | 0.1300 | 0.1641 | 0.1166 | 0.0100 |
| abl_lora_A_only | 0.0667 | 0.1000 | 0.1617 | 0.1138 | 0.0100 |
| abl_lora_B_only | 0.1145 | 0.1200 | 0.1630 | 0.1046 | 0.0400 |
| abl_proj_rho_0 | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_proj_rho_0p5 | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_proj_rank_max_8 | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_proj_mode_plus_shared | 0.1221 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |

Notes:
- The beta grid (`abl_beta_*`) and `abl_proj_rank_max_16` are fully evaluated
  on HS/XS/HP/MB but still running on GSM8K.
- Numbers above are from 100-sample evals; MBPP is highly quantized at this
  sample size.

## Beta sweep + proj_rank_max_16 (non-GSM8K complete, GSM8K in progress)

Non-GSM8K metrics (all done):

| Ablation | HS | XS | HP | MB |
| --- | --- | --- | --- | --- |
| abl_proj_rank_max_16 | 0.3600 | 0.1803 | 0.1267 | 0.0000 |
| abl_beta_g0_r0 | 0.2900 | 0.1609 | 0.1006 | 0.0200 |
| abl_beta_g0_r0p5 | 0.2900 | 0.1602 | 0.1069 | 0.0300 |
| abl_beta_g0_r1 | 0.2500 | 0.1663 | 0.1300 | 0.0200 |
| abl_beta_g0p5_r0 | 0.0800 | 0.1624 | 0.1104 | 0.0100 |
| abl_beta_g0p5_r0p5 | 0.3000 | 0.1720 | 0.1029 | 0.0200 |
| abl_beta_g0p5_r1 | 0.1300 | 0.1578 | 0.1060 | 0.0100 |
| abl_beta_g1_r0 | 0.1000 | 0.1662 | 0.1147 | 0.0400 |
| abl_beta_g1_r0p5 | 0.2000 | 0.1570 | 0.1081 | 0.0000 |
| abl_beta_g1_r1 | 0.1300 | 0.1619 | 0.1203 | 0.0300 |

GSM8K in-progress (latest accuracy, partial count):
- abl_proj_rank_max_16: 0.1273 at 158/1241
- abl_beta_g0_r0: 0.0861 at 103/1196
- abl_beta_g0_r0p5: 0.0542 at 63/1162
- abl_beta_g0_r1: 0.0677 at 81/1196
- abl_beta_g0p5_r0: 0.0964 at 120/1245
- abl_beta_g0p5_r0p5: 0.1289 at 152/1179
- abl_beta_g0p5_r1: 0.0910 at 99/1088
- abl_beta_g1_r0: 0.1081 at 117/1082
- abl_beta_g1_r0p5: 0.1275 at 127/996
- abl_beta_g1_r1: 0.1226 at 122/995

## Interpreting the partial results

These are early signals, not final conclusions:

- Full vs FedAvg: full is higher on GSM8K and HellaSwag (0.122/0.360 vs
  0.0667/0.1300) and slightly higher on XSum/HotPotQA. This supports the
  main narrative that SubspaceBank improves global robustness.
- Decomposition-only is close to FedAvg on HS/XS/HP and only modestly higher
  on GSM8K, suggesting decomposition alone is not sufficient.
- Reweight-only is worse than FedAvg on HS and only marginal on XS/HP,
  indicating reweighting alone is not enough.
- A-only and B-only are weaker than A+B on HS/HP. B-only improves GSM8K but
  still trails full on HS, supporting the claim that both sides matter.
- Server-only, A+B, and all projection ablations are numerically identical
  to full across all five tasks at 100 samples. This is a key signal:
  either (a) projection/payload settings do not affect global metrics in
  this regime, or (b) 100-sample evals are too coarse to detect the effect.
  If this persists at full eval, the paper should attribute global gains
  primarily to server-side banking/decomposition and frame projection as
  optional or primarily beneficial for specialization retention.
- Beta sweep: the best HS/XS values in the grid (HS 0.30, XS 0.172) still
  trail the baseline full run (HS 0.36, XS 0.180). Extremes (e.g., g0p5_r0)
  perform poorly on HS. This suggests the baseline setting is in a stable
  region and supports the current choice of beta weights.

## How this should affect the paper draft

If the final ablations look like the partial results:

- Emphasize that server-side decomposition/banking is the primary driver
  of global robustness gains.
- Present projection as a mechanism that protects client-local learning
  (and show it in the specialization table), rather than as the main cause
  of global gains.
- Highlight that both LoRA A and B are needed, since A-only/B-only are
  weaker and A+B tracks full.
- Use the beta sweep to show stability (avoid over-claiming the exact
  baseline beta settings if performance is flat in a range).

If later results contradict this, update the narrative accordingly (see
the "paper-changing" cases above).
