import logging
from typing import Dict, List, Tuple

import torch

from federatedscope.core.aggregators.clients_avg_aggregator import \
    ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.llm.algo import (chunked_discrimination, maybe_clear_cuda,
                                     select_target_params,
                                     qr_basis_from_concat)

logger = logging.getLogger(__name__)

_ROBUST_DELTA_RULES = {
    'krum', 'normbounding', 'median', 'trimmedmean', 'bulyan'
}


class UnlearnFedAvgAggregator(ClientsAvgAggregator):
    """FedAvg variant with UNLEARN-style subspace discrimination."""
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model=model, device=device, config=config)
        self._device = torch.device(device) if not isinstance(
            device, torch.device) else device
        self._last_bases: Dict[str, torch.Tensor] = {}

    @property
    def latest_bases(self) -> Dict[str, torch.Tensor]:
        """Return the latest Q bases to broadcast to clients."""
        return self._last_bases

    def aggregate(self, agg_info: Dict) -> Dict[str, torch.Tensor]:
        if not self.cfg.aggregator.unlearn.enable:
            self._last_bases = {}
            return super().aggregate(agg_info)

        if self.cfg.federate.use_ss:
            logger.warning('UNLEARN aggregation is incompatible with secret '
                           'sharing; falling back to vanilla FedAvg.')
            self._last_bases = {}
            return super().aggregate(agg_info)

        models = agg_info.get('client_feedback', [])
        if not models:
            self._last_bases = {}
            return {}

        sample_sizes, client_states = self._normalize_client_states(models)
        base_result = super().aggregate(agg_info)
        weights = self._normalize_weights(sample_sizes)

        global_state = self.model.state_dict()
        cached_global = {
            name: param2tensor(param).detach().to(self._device)
            for name, param in global_state.items()
        }

        delta_mode = self._client_sends_delta()
        client_deltas = self._compute_client_deltas(client_states,
                                                    cached_global, delta_mode)

        target_keys = self._collect_target_keys(client_deltas)
        if not target_keys:
            self._last_bases = {}
            return base_result

        chunk_rows = self.cfg.aggregator.unlearn.chunk_rows
        proj_dtype = self.cfg.aggregator.unlearn.proj_dtype
        mode = self.cfg.aggregator.unlearn.mode.lower()
        lambda_shrink = self.cfg.aggregator.unlearn.lambda_shrink
        beta_shared = self.cfg.aggregator.unlearn.beta_shared
        alpha = self.cfg.aggregator.unlearn.alpha_global

        relevant_deltas = {key: client_deltas[key] for key in target_keys}
        unique_parts, shared_parts = self._discriminate(
            relevant_deltas, chunk_rows, proj_dtype)
        updated_tensors = {}
        stats = []
        bases_payload = {} if self.cfg.aggregator.unlearn.send_Q_to_clients \
            else None

        for key in target_keys:
            base_tensor = cached_global.get(key)
            if base_tensor is None:
                continue

            uniques = unique_parts[key]
            shareds = shared_parts[key]
            if len(uniques) == 0:
                continue

            tilde = self._combine_unique_shared(uniques, shareds, weights,
                                                mode, lambda_shrink,
                                                beta_shared, proj_dtype)
            aggregated_delta = self._weighted_sum(
                tilde, weights, proj_dtype).to(dtype=base_tensor.dtype)
            updated_tensor = (base_tensor + alpha * aggregated_delta).to(
                dtype=global_state[key].dtype, device=global_state[key].device)
            updated_tensors[key] = updated_tensor

            key_stats = self._collect_stats(key, uniques, shareds)
            stats.append(key_stats)

            if bases_payload is not None:
                proj_torch_dtype = self._resolve_proj_dtype(proj_dtype)
                stacked = torch.cat(
                    [delta.to(self._device) for delta in relevant_deltas[key]],
                    dim=0).to(dtype=proj_torch_dtype)
                basis = qr_basis_from_concat(stacked, proj_torch_dtype)
                if basis is not None:
                    bases_payload[key] = basis.cpu()

        if bases_payload is not None:
            self._last_bases = bases_payload
        else:
            self._last_bases = {}

        if stats:
            if getattr(self.cfg, 'wandb', None) and self.cfg.wandb.use:
                try:
                    import wandb
                    round_idx = agg_info.get('round')
                    log_payload = {}
                    for record in stats:
                        sanitized = record['key'].replace('.', '/')
                        perp_norm = record['perp_norm']
                        parallel_norm = record['parallel_norm']
                        ratio = perp_norm / max(parallel_norm, 1e-12)
                        base_tag = f'unlearn/{sanitized}'
                        log_payload[f'{base_tag}/perp_norm'] = perp_norm
                        log_payload[
                            f'{base_tag}/parallel_norm'] = parallel_norm
                        log_payload[f'{base_tag}/perp_parallel_ratio'] = ratio
                    if log_payload:
                        wandb.log(log_payload, step=round_idx)
                except ImportError:
                    logger.warning(
                        "cfg.wandb.use=True but wandb is not installed; skip "
                        "logging UNLEARN stats to wandb.")
                except Exception as exc:
                    logger.warning("Failed to log UNLEARN stats to wandb: %s",
                                   exc)
            for record in stats:
                logger.info(
                    '[UNLEARN] key=%s ||perp||_F=%.4e ||parallel||_F='
                    '%.4e mode=%s alpha=%.3f lambda=%.3f beta=%.3f '
                    'chunk=%d dtype=%s device=%s', record['key'],
                    record['perp_norm'], record['parallel_norm'], mode, alpha,
                    lambda_shrink, beta_shared, chunk_rows, proj_dtype,
                    self._device)

        maybe_clear_cuda(self._device)

        base_result.update(updated_tensors)
        return base_result

    def _normalize_client_states(
        self, models: List[Tuple[int, Dict]]
    ) -> Tuple[List[int], List[Dict[str, torch.Tensor]]]:
        sample_sizes, client_states = [], []
        for sample_size, payload in models:
            sample_sizes.append(sample_size)
            if isinstance(payload, list):
                if len(payload) != 1:
                    raise ValueError(
                        'UNLEARN aggregator only supports single model case.')
                payload = payload[0]

            norm_state = {}
            for name, value in payload.items():
                tensor = param2tensor(value)
                if not isinstance(tensor, torch.Tensor):
                    continue
                tensor = tensor.detach().to(self._device)
                norm_state[name] = tensor
            client_states.append(norm_state)
        return sample_sizes, client_states

    def _normalize_weights(self, sample_sizes: List[int]) -> List[float]:
        num_clients = len(sample_sizes)
        if num_clients == 0:
            return []

        if self.cfg.federate.ignore_weight:
            weight = 1.0 / num_clients
            return [weight for _ in sample_sizes]

        total = sum(sample_sizes)
        if total <= 0:
            weight = 1.0 / num_clients
            return [weight for _ in sample_sizes]

        return [size / total for size in sample_sizes]

    def _client_sends_delta(self) -> bool:
        if self.cfg.asyn.use:
            return True
        return self.cfg.aggregator.robust_rule in _ROBUST_DELTA_RULES

    def _compute_client_deltas(
            self, client_states: List[Dict[str, torch.Tensor]],
            cached_global: Dict[str, torch.Tensor],
            delta_mode: bool) -> Dict[str, List[torch.Tensor]]:
        deltas_ordered: Dict[str, List[torch.Tensor]] = {}
        with torch.no_grad():
            all_keys = set(cached_global.keys())
            for state in client_states:
                all_keys.update(state.keys())

            for name in all_keys:
                reference = cached_global.get(name)
                if reference is None:
                    continue
                per_client: List[torch.Tensor] = []
                ref_tensor = reference.to(device=self._device)
                for state in client_states:
                    client_tensor = state.get(name)
                    if client_tensor is None:
                        tensor = ref_tensor.clone()
                    else:
                        tensor = client_tensor.to(device=self._device,
                                                  dtype=ref_tensor.dtype)
                    if not torch.isfinite(tensor).all():
                        logger.warning('Detected non-finite tensor in client '
                                       'update for %s; substituting reference '
                                       'weights.', name)
                        tensor = ref_tensor.clone()
                    if delta_mode:
                        per_client.append(tensor)
                    else:
                        per_client.append(tensor - ref_tensor)
                if per_client:
                    deltas_ordered[name] = per_client
        return deltas_ordered

    def _collect_target_keys(
            self, client_deltas: Dict[str, List[torch.Tensor]]) -> List[str]:
        only_lora = self.cfg.aggregator.unlearn.only_lora
        target_modules = self.cfg.aggregator.unlearn.target_modules
        keys = []
        for name, tensors in client_deltas.items():
            if not tensors or tensors[0].ndim != 2:
                continue
            template = tensors[0]
            descriptor = {name: template}
            selected = select_target_params(descriptor, only_lora,
                                            target_modules)
            if selected:
                keys.append(name)
        return keys

    def _discriminate(
        self,
        deltas: Dict[str, List[torch.Tensor]],
        chunk_rows: int,
        proj_dtype: str,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        return chunked_discrimination(deltas, chunk_rows, proj_dtype)

    def _combine_unique_shared(self, uniques: List[torch.Tensor],
                               shareds: List[torch.Tensor],
                               weights: List[float], mode: str,
                               lambda_shrink: float, beta_shared: float,
                               proj_dtype: str) -> List[torch.Tensor]:
        combined = []
        if mode not in {'shrink', 'mean_shared'}:
            logger.warning(
                'Unsupported UNLEARN mode %s; defaulting to '
                '"shrink".', mode)
            mode = 'shrink'

        work_dtype = self._resolve_proj_dtype(proj_dtype)
        uniques_fp = [tensor.to(dtype=work_dtype) for tensor in uniques]
        shareds_fp = [tensor.to(dtype=work_dtype) for tensor in shareds]

        if mode == 'mean_shared':
            shared_mean = torch.zeros_like(shareds_fp[0], dtype=work_dtype)
            for weight, shared in zip(weights, shareds_fp):
                shared_mean = shared_mean + weight * shared
        else:
            shared_mean = None

        for unique_orig, unique_fp, shared_fp in zip(uniques, uniques_fp,
                                                     shareds_fp):
            if mode == 'mean_shared':
                combined_tensor = unique_fp + beta_shared * shared_mean
            else:  # shrink
                combined_tensor = unique_fp + (1.0 - lambda_shrink) * shared_fp
            combined.append(combined_tensor.to(dtype=unique_orig.dtype))
        return combined

    def _weighted_sum(self, tensors: List[torch.Tensor], weights: List[float],
                      proj_dtype: str) -> torch.Tensor:
        if not tensors:
            return torch.tensor(0.0, device=self._device)
        dtype = self._resolve_proj_dtype(proj_dtype)
        acc = torch.zeros_like(tensors[0], dtype=dtype, device=self._device)
        for tensor, weight in zip(tensors, weights):
            acc = acc + tensor.to(device=self._device, dtype=dtype) * weight
        return acc

    def _collect_stats(self, key: str, uniques: List[torch.Tensor],
                       shareds: List[torch.Tensor]) -> Dict[str, float]:
        unique_norms = [
            torch.linalg.norm(tensor.to(device='cpu', dtype=torch.float32))
            for tensor in uniques
        ]
        shared_norms = [
            torch.linalg.norm(tensor.to(device='cpu', dtype=torch.float32))
            for tensor in shareds
        ]
        return {
            'key': key,
            'perp_norm': torch.stack(unique_norms).mean().item(),
            'parallel_norm': torch.stack(shared_norms).mean().item()
        }

    @staticmethod
    def _resolve_proj_dtype(name: str) -> torch.dtype:
        mapping = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float64': torch.float64,
            'fp64': torch.float64,
            'double': torch.float64,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
        }
        return mapping.get(name.lower(), torch.float32)
