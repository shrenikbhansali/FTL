# bank_perclient: Subspace Bank with Per-Client Projections

This document explains the bank_perclient algorithm used in the FTL
pipeline. It is a mathematical description of how the subspaces are
built, how aggregation is performed, how client training is modified,
and how the logged fractions should be interpreted.

The implementation lives in:
- `FTL/federatedscope/core/aggregators/unlearn_fedavg_aggregator.py`
- `FTL/federatedscope/core/workers/server.py`
- `FTL/federatedscope/core/workers/client.py`
- `FTL/federatedscope/llm/trainer/trainer.py`

## 1. Setup and notation

We run federated learning with N clients. Each client fine-tunes a base
LLM using LoRA adapters. The server aggregates client updates every
round.

We focus on a single LoRA parameter matrix (a "key") and treat it as a
2D matrix:

- Key index: k (for example, a LoRA A or B matrix on q_proj/k_proj/etc.)
- Client index: i in {1, ..., N}
- LoRA delta matrix: D_{k,i} in R^{d_out x d_in}

The delta is either:

- D_{k,i} = W_{k,i} - W_k (client weight minus server weight), or
- D_{k,i} provided directly as a delta (async/robust settings).

Only 2D LoRA matrices are processed. Other parameters are aggregated
with standard FedAvg.

Client weights are w_i:

- If `federate.ignore_weight` is false, w_i = n_i / sum_j n_j (n_i is
  client sample size).
- Otherwise, w_i = 1 / N.

## 2. Subspace bank: shared and private bases

The bank decomposes each D_{k,i} into:

1) A shared component that is common across clients.
2) A private component unique to each client.
3) A residual component (what remains).

### 2.1 Build the shared basis S_k

Stack all client deltas vertically:

M_k = concat_i D_{k,i}  in R^{(N * d_out) x d_in}

Compute a thin SVD:

M_k = U_k Sigma_k V_k^T

The columns of V_k are orthonormal and span the row space of M_k.
We choose a rank r_global and define:

S_k = V_k[:, 0:r_global]  in R^{d_in x r_global}

Rank selection:

- If `bank.energy_target > 0`, choose the smallest r_global that reaches
  the desired cumulative singular value energy.
- Otherwise, r_global is the configured `bank.r_global`.

### 2.2 Build client private bases P_{k,i}

For each client i:

1) Remove the shared component:

R_{k,i}^{(0)} = D_{k,i} - (D_{k,i} S_k) S_k^T

2) Compute an SVD on R_{k,i}^{(0)} and take r_client right singular
vectors as a candidate basis.

3) Orthogonalize that basis against S_k (so private directions do not
overlap the shared space). The result is:

P_{k,i} in R^{d_in x r_client}

If r_client <= 0 or R_{k,i}^{(0)} is numerically tiny, P_{k,i} is empty.

## 3. Delta decomposition

Each D_{k,i} is split into three orthogonal components:

Shared component (projection onto S_k):

D_{k,i}^{(glob)} = (D_{k,i} S_k) S_k^T

Private component (projection onto P_{k,i} inside the residual):

D_{k,i}^{(priv)} = (R_{k,i}^{(0)} P_{k,i}) P_{k,i}^T

Residual:

R_{k,i} = D_{k,i} - D_{k,i}^{(glob)} - D_{k,i}^{(priv)}

Interpretation:

- D_{k,i}^{(glob)}: shared directions across clients.
- D_{k,i}^{(priv)}: client-specific directions that are orthogonal to the
  shared space.
- R_{k,i}: leftover energy not captured by the chosen ranks.

## 4. Structured aggregation

Instead of averaging raw deltas, the bank constructs a structured delta:

tilde{D}_{k,i} = D_{k,i}^{(priv)}
               + beta_global * D_{k,i}^{(glob)}
               + beta_resid * R_{k,i}

Then aggregate with FedAvg weights:

Delta_k = sum_i w_i * tilde{D}_{k,i}

Global update for this key:

W_k <- W_k + alpha_global * Delta_k

Notes:

- The update is applied only to target LoRA keys. All other parameters
  use standard FedAvg.
- alpha_global is `aggregator.unlearn.alpha_global`.
- beta_global and beta_resid come from `aggregator.unlearn.bank`.

## 5. Per-client projection bases (the "bank_perclient" part)

The bank also sends each client a basis containing other clients' private
subspaces (optionally plus the shared basis). This is the "bank_perclient"
behavior.

For each client i and key k, build:

Q_k^{(i)} = orth( [P_{k,j}]_{j != i}  (+ S_k if enabled) )

Implementation details:

- Concatenate the other clients' private bases along columns.
- Optionally append S_k if `bank_proj_mode` is "others_plus_shared".
- Orthonormalize with QR.
- Optionally truncate to `bank_proj_rank_max`.

This per-client payload is sent only when:

- `aggregator.unlearn.send_Q_to_clients` is true, and
- `bank.bank_send_per_client` is true.

Clients will ignore it unless `train.unlearn.bank_use_per_client` is
also true.

## 6. Client training loop with gradient projection

The local training loop is standard LLM fine-tuning plus one step:
gradient projection.

Round t:

1) Server broadcasts global model and Q_k^{(i)} bases.
2) Client receives model and stores bases.
3) Client trains for local_update_steps:

Pseudo-code for one batch:

```
loss = model(x)
loss.backward()

for each LoRA parameter W_k with basis Q_k:
    g = grad(W_k)
    projection = (g @ Q_k) @ Q_k^T
    g <- g - strength * projection

clip_gradients()
optimizer.step()
scheduler.step()
```

Projection strength:

strength = proj_rho * schedule(round)

Where:

- proj_rho is `train.unlearn.proj_rho`.
- schedule(round) is optional (linear ramp in `proj_schedule`).
- If DeepSpeed is enabled, projection is disabled by design.

4) Client sends updated model or delta back to server.
5) Server aggregates and repeats.

Effect:

Each client is prevented from moving along subspaces owned by other
clients (and optionally the shared subspace). This reduces destructive
interference while still allowing shared learning through the
beta_global component in aggregation.

## 7. Fraction diagnostics

For each client i and key k, the server computes energy fractions:

global_fraction = ||D_{k,i}^{(glob)}||_F^2 / ||D_{k,i}||_F^2
private_fraction = ||D_{k,i}^{(priv)}||_F^2 / ||D_{k,i}||_F^2
resid_fraction = ||R_{k,i}||_F^2 / ||D_{k,i}||_F^2

Then:

- Per-key values are averaged across clients.
- A global summary is computed by weighting each key by its number of
  parameters (numel).

How to interpret:

- High global_fraction: many updates lie in the shared subspace.
- High private_fraction: clients have strong unique directions.
- High resid_fraction: the chosen ranks are too small or the updates are
  noisy (consider increasing r_global, r_client, or energy_target).

These diagnostics are logged under:

- `unlearn_bank/<key>/global_fraction`, `private_fraction`, `resid_fraction`
- `unlearn_bank/summary/*`

## 8. Parameter map (bank_perclient)

Key knobs used by the algorithm:

- `bank.r_global`: max rank for S_k.
- `bank.energy_target`: if > 0, choose r_global by energy.
- `bank.r_client`: max rank for each P_{k,i}.
- `bank.beta_global`: weight on shared component.
- `bank.beta_resid`: weight on residual component.
- `bank.bank_proj_mode`: "others_private" or "others_plus_shared".
- `bank.bank_proj_rank_max`: optional cap on Q_k^{(i)} rank.
- `train.unlearn.project_grads`: enable projection.
- `train.unlearn.proj_rho`: projection strength.
- `train.unlearn.bank_use_per_client`: consume per-client payloads.

See `FTL/yamls/individual_federated_bank_perclient.yaml` for the concrete
values used in the local pipeline.

## 9. Summary

bank_perclient is FedAvg plus a subspace bank:

- It decomposes each LoRA update into shared, private, and residual parts.
- It aggregates a structured delta that can emphasize or suppress each
  part.
- It sends per-client projection bases so local training avoids stepping
  into other clients' private subspaces.

The result is a more stable multi-client LoRA fine-tuning process that
preserves client-specific specialization while still allowing shared
global progress.
