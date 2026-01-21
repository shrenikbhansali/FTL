# Answers: bank_perclient codebase questions

This document answers the requested questions for the current training
workflow that uses `scripts/run_local_individual_sweep_pipeline.sh`. It
is grounded in the code paths and YAMLs actually used in that sweep.

Scope references:
- Sweep script: `FTL/scripts/run_local_individual_sweep_pipeline.sh`
- Banked configs used by the sweep:
  - `FTL/yamls/individual_federated_participation_bank_perclient.yaml`
  - `FTL/yamls/individual_federated_budget_bank_perclient.yaml`
  - `FTL/yamls/individual_federated_sharded_light_bank_perclient.yaml`
- Aggregator: `FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`
- LoRA target selection: `FTL/federatedscope/llm/algo/unlearn_ops.py`
- Client/server plumbing: `FTL/federatedscope/core/workers/server.py`,
  `FTL/federatedscope/core/workers/client.py`
- Client trainer + projection: `FTL/federatedscope/llm/trainer/trainer.py`

## 1) Banked objects: A vs B vs both vs reconstructed (\Delta W)

### 1.1 Exactly what tensor(s) are treated as D_{k,i} for banking?

- D_{k,i} is the *parameter delta* for each selected parameter key (k).
  In the synchronous case, it is computed as:
  D_{k,i} = W_{k,i} - W_k (client post-local minus server pre-local).
  See `_compute_client_deltas` in
  `FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`.
- There is no reconstruction of \Delta W = B A. The bank operates on the
  raw 2D parameter tensors themselves.
- Because `aggregator.unlearn.only_lora: True` in the sweep YAMLs, the
  selected tensors are the LoRA A and LoRA B matrices (parameters whose
  names include `lora_A` or `lora_B`). This is enforced by
  `select_target_params` in `FTL/federatedscope/llm/algo/unlearn_ops.py`.

Answer: both LoRA A and LoRA B are banked, independently. There is no
bank over the reconstructed BA, and nothing is concatenated.

### 1.2 If both A and B are banked, how?

- Procedure: identical for A and B. Both use the same bank code path.
  The shared basis S_k is built from right singular vectors of the
  stacked deltas (see `_build_global_basis_for_bank`), and the private
  basis P_{k,i} is built from the right singular vectors of the residual.
- Subspace side: always the right-subspace (column space) because the
  SVD is taken on the concatenated matrix and the right singular vectors
  are used. There is no left-subspace logic for B or A.
- Parameters (r_g, r_c, beta_g, beta_r) are shared across all keys. There
  is no separate configuration for A vs B in code or YAML.

### 1.3 What is the “key” k in code?

- The key is the **exact state_dict parameter name** for a 2D tensor.
  It is not an aggregated layer name.
- With `only_lora: True`, k corresponds to each LoRA A or B matrix in
  each module, per layer (for example, a q_proj lora_A weight in a
  specific layer).

### 1.4 Which modules are included by default?

The sweep YAMLs enable LoRA on:

- `target_modules: ["q_proj","k_proj","v_proj","o_proj"]`

Since `only_lora: True`, the bank operates on the LoRA A/B parameters
inside those modules only. No MLP modules (gate/up/down), and no
layer-range filters are applied by default.

### 1.5 Are embedding / lm_head / norms ever included?

- Not for banking in the sweep configs.
- `modules_to_save: ["embed_tokens","lm_head"]` only affects checkpoint
  saving and does not introduce LoRA weights for these modules.
- Norms are 1D, and the bank only operates on 2D tensors.
- With `only_lora: True`, any non-LoRA parameter is excluded regardless.

### 1.6 Are deltas taken for LoRA weights directly, or inferred from optimizer state?

Deltas are taken directly from parameters. There is no use of optimizer
state in the bank. See `_compute_client_deltas` in
`FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`.

**Deliverable (config flags + code paths + shapes)**

Config flags that determine which tensors are banked:
- `aggregator.unlearn.only_lora` (True in sweep YAMLs)
- `aggregator.unlearn.target_modules` (q_proj/k_proj/v_proj/o_proj)
- `llm.adapter.args.target_modules` (LoRA injection sites)

Code paths:
- Selection of keys: `UnlearnFedAvgAggregator._collect_target_keys` ->
  `select_target_params` in `FTL/federatedscope/llm/algo/unlearn_ops.py`
- Delta construction: `UnlearnFedAvgAggregator._compute_client_deltas`
  in `FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`

Shapes of banked tensors:
- LoRA A matrix: shape (r, d_in)
- LoRA B matrix: shape (d_out, r)
- r comes from `llm.adapter.args.r` (r = 8 in sweep YAMLs)
- d_in/d_out are module-specific (set by the base model)

## 2) Participation semantics: S_t vs cached P_{k,j} and cache policy

### 2.1 S_k (shared basis) uses which deltas?

Only deltas from **participating clients in the current round**.
`aggregate()` stacks `client_feedback` received in that round and builds
S_k from those deltas only. There is no history cache for banked S_k.

### 2.2 Private bases P_{k,i} computed for which clients?

Only for clients that participated in the current round. There is no
storage or reuse of private bases across rounds in the bank path.

### 2.3 Q_k^{(i)} uses which “other clients”?

Only other **participating clients in the same round**. The payload
construction uses the `private_bases` dictionary built in the current
round; no other cached clients are included.

### 2.4 Cache lifetime / EMA / invalidation?

Not applicable for bank_perclient. There is **no cache or TTL** for
banked bases. The EMA cache (`_ema_cache`) exists only in the non-bank
UNLEARN path and is not used when `bank.enable` is true.

### 2.5 Do inactive clients receive any broadcast payload?

No. The server only sends model parameters (and any bases) to the
clients selected by the sampler in that round (see
`broadcast_model_para` in `FTL/federatedscope/core/workers/server.py`).

### 2.6 Is client sampling deterministic given seed?

Sampling uses `np.random.choice` in `UniformSampler`
(`FTL/federatedscope/core/sampler.py`). Seeding is set by
`setup_seed(cfg.seed)` in `federatedscope/main.py`. So the sampling is
deterministic given a fixed seed and deterministic runtime. The sampled
set S_t is **not explicitly logged** in the codebase.

### 2.7 Is the bank computed per-round or incrementally?

Per-round only. The bank is computed inside `aggregate()` after the
server has collected the client feedback for that round. It is not
updated incrementally as deltas arrive.

### 2.8 Any privacy constraints that restrict per-client delta access?

If secret sharing is enabled (`cfg.federate.use_ss`), UNLEARN is
disabled and the bank path does not run. Otherwise, the bank uses raw
per-client deltas with no additional privacy constraints.

**Deliverable (3-5 sentences, S_t and Q^{(i)})**

S_t is the set of clients sampled in the current round by the server
sampler. The shared basis S_k and all private bases P_{k,i} are computed
only from deltas provided by clients in S_t for that round. The per-
client payload Q^{(i)} is built from the private bases of other clients
in S_t (optionally plus S_k), and is sent only to participating clients.
There is no cache/TTL/EMA for bank_perclient; bases are recomputed fresh
each round and discarded after broadcast.

## 3) Projection plumbing: where projection occurs relative to clipping and Adam

### 3.1 What exactly is projected?

The **raw gradient** (`param.grad`) for each banked LoRA parameter. The
projection is applied right after `loss.backward()` and before gradient
clipping and optimizer step.

### 3.2 What optimizer is used by default?

The sweep YAMLs set:
- `train.optimizer.type: Adam`
- `lr: 2.0e-4`, `betas: [0.9, 0.95]`, `eps: 1e-5`

The optimizer is created in `get_optimizer` with no explicit param
groups; it uses `model.parameters()` directly.

### 3.3 Are moments projected?

No. The projection is applied only to the instantaneous gradients. There
is no code that projects Adam moments (m_t or v_t).

### 3.4 Where does gradient clipping happen relative to projection?

Projection happens first, then clipping:
`loss.backward()` -> `_project_gradients()` -> `clip_grad_norm_()` ->
`optimizer.step()`.

### 3.5 Is projection applied every local step?

Yes. `_project_gradients` is called in `_hook_on_batch_backward` for
every batch step.

### 3.6 Is projection applied to all LoRA keys or only the banked subset?

Only to the subset for which bases are provided. The trainer checks
`self._unlearn_bases` and applies projection only when a basis exists
for that parameter name. With `only_lora: True` and bank enabled, this
typically includes all LoRA A/B parameters for q/k/v/o.

### 3.7 Does projection apply to the same tensors that are communicated?

Communication sends the full model (or its deltas), but projection only
applies to the banked subset. If a parameter is trained but not banked,
its gradient is not projected.

### 3.8 Is rho constant or scheduled?

By default, rho is constant: `train.unlearn.proj_rho: 1.0`. There is an
optional linear schedule via `train.unlearn.proj_schedule`, which
rescales projection strength by round, but it is not set in the sweep
YAMLs.

### 3.9 Numerical details

- Q is stored in CPU memory and cast to float32 in
  `LLMTrainer.set_unlearn_bases`.
- The projection uses `proj_dtype` from the payload (default float32).
- Formula is exactly: `g <- g - strength * (g @ Q) @ Q^T`.

**Deliverable (code location and order)**

Projection is applied in `LLMTrainer._project_gradients` within
`FTL/federatedscope/llm/trainer/trainer.py`, called from
`LLMTrainer._hook_on_batch_backward`. The exact order is:
`loss.backward()` -> `self._project_gradients(ctx)` ->
`clip_grad_norm_()` -> `optimizer.step()` -> `scheduler.step()`.

## 4) Recompute frequency: every round vs every m, and EMA

### 4.1 How often is the bank recomputed?

Every round. The banked bases are built inside
`UnlearnFedAvgAggregator.aggregate()` for each aggregation step when
`bank.enable` is true.

### 4.2 Intermediate rounds?

Not applicable. There is no reuse of previous bank state between rounds.

### 4.3 EMA smoothing?

No EMA is used in the bank_perclient path. EMA is only present in the
non-bank UNLEARN path (`broadcast.ema_gamma` and `_ema_cache`).

### 4.4 Warm-up phase?

No warm-up is implemented for bank_perclient. There are no config
entries to delay bank activation.

### 4.5 Recompute per key or selective?

All eligible keys are processed each round. If deltas are missing or too
small, the corresponding basis may be None, but there is no explicit
energy threshold to skip keys beyond that.

### 4.6 Recompute frequency different for server-only vs bank_perclient?

No. The bank_perclient path runs inside the bank-enabled aggregate call
and recomputes every round.

### 4.7 Do you log and persist bank components per round?

Only scalar stats (fractions) are logged. Bases are not persisted to
disk; they are used to form per-client payloads and then discarded.

**Deliverable (config values + code trigger)**

There are no config values for recompute frequency, warm-up, or EMA in
the bank path. Recompute is triggered in
`UnlearnFedAvgAggregator.aggregate()` each round when
`aggregator.unlearn.bank.enable: True` (set in the sweep YAMLs).

## 5) Rank constraints and payload caps

### 5.1 How are ranks chosen in practice?

- r_global is fixed by `bank.r_global` unless `bank.energy_target > 0`.
- If `bank.energy_target > 0`, r_global is chosen by SVD energy.
- r_client is fixed by `bank.r_client`; no energy-based selection exists
  for private bases.

Sweep YAML values:
- `r_global: 1`
- `r_client: 12`
- `energy_target: 0.0`
- `bank_proj_rank_max: 0` (no cap)

Default config values (if YAML does not override) in
`FTL/federatedscope/core/configs/cfg_aggregator.py`:
- `r_global: 4`, `r_client: 4`, `energy_target: 0.0`,
  `bank_proj_rank_max: 0`

### 5.2 Energy definition

Energy is defined using squared singular values:
sum_{j<=r} sigma_j^2 / sum_j sigma_j^2. This is used in
`_build_global_basis_for_bank`.

### 5.3 Hard caps

- r_global is capped by the number of available columns (d_in).
- r_client is capped by the available private basis rank after
  orthogonalization.
- r_Q (payload rank cap) is `bank_proj_rank_max`; if > 0, the final Q is
  truncated to that many columns after QR.

### 5.4 What if r_g + r_c > d_in?

There is no explicit r_g + r_c constraint. The implementation implicitly
caps ranks:
- r_global is capped by d_in.
- r_client is capped by the residual basis rank and reduced after
  orthogonalization against S_k.
So if r_g + r_c exceeds d_in, the private basis is simply reduced to the
available orthogonal directions.

### 5.5 Private basis skip rules

Private SVD is skipped if:
- `r_client <= 0`, or
- residual norm <= 1e-8 * max(||delta||, 1e-12).

This threshold is in `_decompose_with_bank`.

### 5.6 Payload selection policy when rank is capped

Q^{(i)} is constructed by concatenating all other clients' private bases
for that round (and optionally S_k), then orthonormalizing via QR. If a
rank cap is set, the first `rank_max` columns of Q are kept. There is no
energy-based or recency-based selection.

### 5.7 Does payload include shared basis S_k by default?

No. In the sweep YAMLs `bank_proj_mode: "others_private"`, so S_k is not
included. If set to "others_plus_shared", S_k is included and then
subject to the same rank cap.

### 5.8 Orthonormalization after truncation/concatenation

Yes. The payload basis is orthonormalized with QR in
`_orthonormalize_basis`. Truncation to `rank_max` happens after QR.

### 5.9 Do ranks differ by key/layer automatically?

r_global can vary per key if energy_target is used. r_client is fixed
but may effectively be smaller for some keys based on residual rank.

**Deliverable (config defaults + function constructing Q^{(i)})**

Config defaults are in `FTL/federatedscope/core/configs/cfg_aggregator.py`
and overridden in the sweep YAMLs listed above. The Q^{(i)} construction
function is `_build_bank_per_client_payload` in
`FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`.
