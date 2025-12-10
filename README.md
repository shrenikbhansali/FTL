# FederatedScope-LLM with SubspaceBank via UNLEARN

We extend FederatedScope-LLM (FS-LLM) with a **subspace bank** designed to keep multi-task LoRA fine-tuning stable. The bank augments UNLEARN-style aggregation (already present in upstream FS-LLM) by learning a shared subspace plus per-client private directions for every LoRA key, then feeding those directions back to the clients so they avoid stepping on each other.

---

## Table of Contents

1. [Subspace Bank: Concept, Math, and Implementation](#subspace-bank-concept-math-and-implementation)
2. [Configuration Guide](#configuration-guide)
3. [Running Training & Evaluation Jobs](#running-training--evaluation-jobs)
4. [Artifacts, Logging, and Monitoring](#artifacts-logging-and-monitoring)
5. [Troubleshooting Checklist](#troubleshooting-checklist)
6. [Code Changes vs. Upstream FederatedScope](#code-changes-vs-upstream-federatedscope)

---

## Subspace Bank: Concept, Math, and Implementation

SubspaceBank introduces three components:

1. **Shared vs. private delta decomposition** (server side).
2. **Structured aggregation of those components** (server side).
3. **Client-side gradient projection using everyone else’s private bases** (trainer side).

All operations target LoRA matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.). Let `D_{k,i} ∈ ℝ^{d_out×d_in}` be client `i`’s LoRA delta for key `k`.

### Step 1 – Build a shared basis `S_k`
1. Stack all client deltas vertically: `M_k = concat_i D_{k,i}`.
2. Compute a thin SVD: `M_k = U_k Σ_k V_k^⊤`.
3. Pick a rank `r_global` (fixed or by cumulative energy). Use the first `r_global` columns of `V_k` as `S_k` (orthonormal).

This captures directions present across clients (similar to classic UNLEARN but per key).

### Step 2 – Build private bases `P_{k,i}`
For each client:
1. Remove the shared component: `R_{k,i}^{(0)} = D_{k,i} - (D_{k,i} S_k) S_k^⊤`.
2. Compute a rank-`r_client` basis from `R_{k,i}^{(0)}` via SVD, then orthogonalize it against `S_k`. The result is `P_{k,i}`.

These bases model client-specific skills (code-only patterns, GSM8K math heuristics, etc.) inside the orthogonal complement of the shared subspace.

### Step 3 – Decompose each delta
\[
D_{k,i}^{(\text{glob})} = (D_{k,i} S_k) S_k^⊤,\qquad
D_{k,i}^{(\text{priv})} = (R_{k,i}^{(0)} P_{k,i}) P_{k,i}^⊤,\qquad
R_{k,i} = D_{k,i} - D_{k,i}^{(\text{glob})} - D_{k,i}^{(\text{priv})}.
\]

### Step 4 – Structured aggregation
Instead of averaging raw deltas, we aggregate
\[
\tilde D_{k,i} = D_{k,i}^{(\text{priv})} + \beta_{\text{glob}} D_{k,i}^{(\text{glob})} + \beta_{\text{resid}} R_{k,i}.
\]
The server forms `ΔW_k = Σ_i w_i \tilde D_{k,i}` and updates `W_k` accordingly (with global learning rate `α_global`).

`β_{glob}` controls how much of the shared signal survives; `β_{resid}` lets you mix in orthogonal leftovers (usually 0).

### Step 5 – Client-side gradient projection
For each client `i`, the server concatenates every **other** client’s private bases (and, optionally, `S_k`) to build
\[
Q_k^{(i)} = \text{orth}\big([P_{k,j}]_{j ≠ i} \cup (\text{include } S_k?)\big).
\]
During local fine-tuning the trainer replaces every gradient `g_{k,i}` with
\[
 g_{k,i} \leftarrow g_{k,i} (I - ρ Q_k^{(i)} Q_k^{(i)⊤}),
\]
which removes motion inside subspaces owned by other clients. `ρ = train.unlearn.proj_rho` scales the projection strength (default 1.0). This keeps client updates inside the span of the shared basis plus their own private basis, reducing interference.

### Diagnostics
For each key we log
* `bank/global_fraction = ‖D_{k,i}^{(glob)}‖_F² / ‖D_{k,i}‖_F²`
* `bank/private_fraction`
* `bank/resid_fraction`

These tell you how much energy the current ranks capture.

---

## Configuration Guide

The snippet below shows the essential YAML blocks (see `yamls/full/llama2_composite_meta3_unlearn_bank_perclient.yaml` for the full config):

```yaml
aggregator:
  type: "unlearn_fedavg"
  unlearn:
    enable: True
    mode: "shrink"
    alpha_global: 1.0
    chunk_rows: 4096
    send_Q_to_clients: True          # broadcast per-client bases
    bank:
      enable: True
      r_global: 4                    # shared basis rank (or 0 + energy_target)
      r_client: 4
      beta_global: 1.0
      beta_resid: 0.0
      energy_target: 0.0             # set >0 to auto-pick r_global via SVD energy
      bank_send_per_client: True     # expose per-client payloads
      bank_proj_mode: "others_private"  # or "others_plus_shared"
      bank_proj_rank_max: 0          # optional truncation

train:
  local_update_steps: 30
  unlearn:
    project_grads: True
    bank_use_per_client: True        # consume per-client payloads
    proj_rho: 1.0                    # gradient projection strength
```

### Hyperparameter cheatsheet

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `bank.r_global` | Max rank of shared basis `S_k` | 2–8 |
| `bank.energy_target` | Fraction of SVD energy captured (>0 overrides `r_global`) | 0.9 |
| `bank.r_client` | Max rank of `P_{k,i}` | 2–8 |
| `bank.beta_global` | Weight on shared component | 1.0 |
| `bank.beta_resid` | Weight on residual | 0.0 or 0.1 |
| `bank.bank_proj_mode` | Whether to include `S_k` in `Q_k^{(i)}` | `others_private` or `others_plus_shared` |
| `bank.bank_proj_rank_max` | Hard cap on broadcast basis rank | 0 (no cap) |
| `train.unlearn.proj_rho` | Gradient projection strength | 1.0 |

Everything else (optimizer, datasets, wandb, etc.) follows stock FS-LLM.

---

## Running Training & Evaluation Jobs

### Environment setup
```bash
conda create -n fs-llm python=3.9
conda activate fs-llm
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .[llm]
```

### Training
```bash
python federatedscope/main.py --cfg yamls/full/llama2_composite_meta3_unlearn_bank_perclient.yaml
```
Artifacts:
* Global ckpt – `ckpts/full/llama2_composite_meta3_unlearn_bank_perclient.ckpt`
* Client adapters – `ckpts/.../clients/client_{1,2,3}/client_{1,2,3}.ckpt`

### Slurm batch helpers

1. **Train + eval in one shot**
   ```bash
   sbatch scripts/sbatch_train_eval_llama_unlearn_bank_perclient_full.sbatch
   ```
2. **Evaluate an existing checkpoint**
   ```bash
   bash scripts/run_llama_bank_perclient_followups.sh
   ```
   This submits:
   * `scripts/sbatch_eval_llama_unlearn_bank_perclient_global.sbatch`
   * `scripts/sbatch_eval_llama_unlearn_bank_perclient_clients.sbatch`

### Notebook analysis
Run `full_pipeline_accuracy_review.ipynb` after evaluations complete. Section 5 compares the bank variant to Standard, UNLEARN, and UNLEARN-LOO using metrics from `eval_result/` and plots stored in `results_full/`.

---

## Artifacts, Logging, and Monitoring

| Artifact | Location |
|----------|----------|
| Global checkpoint | `ckpts/full/llama2_composite_meta3_unlearn_bank_perclient.ckpt` |
| Client checkpoints | `ckpts/.../clients/client_i/client_i.ckpt` |
| Eval configs/logs | `results_full/{global,clients}/llama2_unlearn_bank_perclient/...` |
| Accuracy JSONs | `eval_result/accuracies_ckpts_full_llama2_composite_meta3_unlearn_bank_perclient*.json` |
| Wandb project | `FTL-Dev` (entity `sbhansali8-georgia-institute-of-technology`) |

Logging highlights:
* Aggregator prints classic UNLEARN stats (`||perp||`, ranks, energies) **and** bank fractions.
* Clients emit warnings if gradient projection is disabled (e.g., when DeepSpeed is on).
* Wandb mirrors all metrics for later comparison.

---

## Troubleshooting Checklist

1. **No bank stats appearing** – Verify `aggregator.unlearn.bank.enable=True`. Otherwise the code silently falls back to legacy UNLEARN.
2. **Clients ignoring projections** – Ensure `aggregator.unlearn.send_Q_to_clients=True`, `bank.bank_send_per_client=True`, and `train.unlearn.bank_use_per_client=True`.
3. **Missing accuracy rows in the notebook** – Re-run `bash scripts/run_llama_bank_perclient_followups.sh`; pending entries show up as `NaN`/“pending”.
4. **Residual energy too high** – Increase `r_global`, `r_client`, or set `bank.energy_target` to something like `0.95`.
5. **Client specialization too weak** – Lower `beta_global` or `proj_rho` so private components dominate.
6. **Training diverges** – Reduce local learning rate or `alpha_global`; the bank does not change the optimizer.

---

## Code Changes vs. Upstream FederatedScope

* **Aggregator core** – `federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`
  * Adds the subspace bank decomposition, structured aggregation, diagnostic logging, and exposes per-client bases via `latest_bases_per_client`.
* **Client/server plumbing** – `federatedscope/core/workers/server.py`, `federatedscope/core/workers/client.py`
  * Server packages `bank_per_client` payloads; clients filter and forward them to the trainer when enabled.
* **Trainer** – `federatedscope/llm/trainer/trainer.py`
  * Stores received bases, applies gradient projection with configurable `proj_rho`, and supports clearing bases when payloads are absent.
* **Configuration schema** – `federatedscope/core/configs/cfg_aggregator.py`, `cfg_training.py`
  * Introduce `aggregator.unlearn.bank.*`, `train.unlearn.bank_use_per_client`, and `train.unlearn.proj_rho`.
* **Experiment assets** – new YAML (`yamls/full/llama2_composite_meta3_unlearn_bank_perclient.yaml`), Slurm scripts (`scripts/sbatch_*bank_perclient*.sbatch`), and helper (`scripts/run_llama_bank_perclient_followups.sh`).
* **Analysis notebook** – `full_pipeline_accuracy_review.ipynb` now plots the bank variant.
