# Overhead results (H200)

Run tag: `20260127_101002`

Configs:
- FedAvg: `FTL/yamls/individual_federated_fedavg.yaml`
- SubspaceBank (server-only): `FTL/yamls/individual_federated_bank_perclient.yaml` with overrides `train.unlearn.project_grads=False`, `aggregator.unlearn.send_Q_to_clients=False`

Notes:
- `total_flops` is only non-zero if `eval.count_flops=True` was set during the run.
- `sys_avg`/`sys_std` are aggregated over all workers (server + clients).
- `e2e_total_flops` below adds SubspaceBank server aggregation FLOPs to the model FLOPs reported in `system_metrics.log`.

## FedAvg

| scope | walltime (min) | total_flops | e2e_total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| server | 139.649 | 0 | 0 | 0 | 23319688 |
| sys_avg | 139.207 | 24.48P | 24.48P | 0.0 | 8.02M |
| sys_std | 0.328 | 34.4P | 34.4P | 0.0 | 6.36M |

### Per-client metrics

| client_id | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| 1 | 139.525 | 38480440074240000.000 | 0 | 5423040 |
| 2 | 139.322 | 8363237621760000.000 | 0 | 5423040 |
| 3 | 139.119 | 109389028515840000.000 | 0 | 5423040 |
| 4 | 138.916 | 5669002506240000.000 | 0 | 5423040 |
| 5 | 138.713 | 3477938688000000.000 | 0 | 5423040 |

## SubspaceBank (server-only)

| scope | walltime (min) | total_flops | e2e_total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| server | 138.803 | 0 | 3.98e11 | 0 | 23319688 |
| sys_avg | 138.379 | 24.48P | 2.448039e16 | 0.0 | 8.02M |
| sys_std | 0.322 | 34.4P | 34.4P | 0.0 | 6.36M |

### Per-client metrics

| client_id | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| 1 | 138.697 | 38480440074240000.000 | 0 | 5423040 |
| 2 | 138.496 | 8363237621760000.000 | 0 | 5423040 |
| 3 | 138.294 | 109389028515840000.000 | 0 | 5423040 |
| 4 | 138.093 | 5669002506240000.000 | 0 | 5423040 |
| 5 | 137.892 | 3477938688000000.000 | 0 | 5423040 |

Raw logs:
- FedAvg: `/home/heck2/sbhansali8/FTL/overhead/20260127_101002/fedavg/overhead_fedavg/system_metrics.log`
- SubspaceBank (server-only): `/home/heck2/sbhansali8/FTL/overhead/20260127_101002/subspacebank_serveronly/overhead_subspacebank_serveronly/system_metrics.log`

## Server-side aggregation FLOPs (SubspaceBank only)

These are estimated FLOPs for SubspaceBank’s server aggregation (SVD/QR + projection matmuls), logged per round from `server_flops.log`.

| metric | value |
| --- | --- |
| rounds logged | 60 |
| total server FLOPs | 398,028,963,840 |
| avg server FLOPs / round | 6,633,816,064 |
| total SVD FLOPs | 307,421,511,680 |
| total QR FLOPs | 16,116,613,120 |
| total matmul FLOPs | 74,490,839,040 |

Raw server FLOPs log:
- SubspaceBank (server-only): `/home/heck2/sbhansali8/FTL/overhead/20260127_101002/subspacebank_serveronly/overhead_subspacebank_serveronly/server_flops.log`
