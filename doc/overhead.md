# Overhead results (H200)

Run tag: `20260127_013234`

Configs:
- FedAvg: `FTL/yamls/individual_federated_fedavg.yaml`
- SubspaceBank (server-only): `FTL/yamls/individual_federated_bank_perclient.yaml` with overrides `train.unlearn.project_grads=False`, `aggregator.unlearn.send_Q_to_clients=False`

Notes:
- `total_flops` is only non-zero if `eval.count_flops=True` was set during the run.
- `sys_avg`/`sys_std` are aggregated over all workers (server + clients).

## FedAvg

| scope | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| server | 136.383 | 0 | 0 | 23319688 |
| sys_avg | 135.935 | 24.48P | 0.0 | 8.02M |
| sys_std | 0.327 | 34.4P | 0.0 | 6.36M |

### Per-client metrics

| client_id | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| 1 | 136.247 | 38480440074240000.000 | 0 | 5423040 |
| 2 | 136.046 | 8363237621760000.000 | 0 | 5423040 |
| 3 | 135.846 | 109389028515840000.000 | 0 | 5423040 |
| 4 | 135.645 | 5669002506240000.000 | 0 | 5423040 |
| 5 | 135.445 | 3477938688000000.000 | 0 | 5423040 |

## SubspaceBank (server-only)

| scope | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| server | 141.581 | 0 | 0 | 23319688 |
| sys_avg | 141.138 | 24.48P | 0.0 | 8.02M |
| sys_std | 0.325 | 34.4P | 0.0 | 6.36M |

### Per-client metrics

| client_id | walltime (min) | total_flops | upload_bytes | download_bytes |
| --- | --- | --- | --- | --- |
| 1 | 141.448 | 38480440074240000.000 | 0 | 5423040 |
| 2 | 141.249 | 8363237621760000.000 | 0 | 5423040 |
| 3 | 141.049 | 109389028515840000.000 | 0 | 5423040 |
| 4 | 140.850 | 5669002506240000.000 | 0 | 5423040 |
| 5 | 140.650 | 3477938688000000.000 | 0 | 5423040 |

Raw logs:
- FedAvg: `/home/heck2/sbhansali8/FTL/overhead/20260127_013234/fedavg/overhead_fedavg/system_metrics.log`
- SubspaceBank (server-only): `/home/heck2/sbhansali8/FTL/overhead/20260127_013234/subspacebank_serveronly/overhead_subspacebank_serveronly/system_metrics.log`
