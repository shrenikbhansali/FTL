import logging
from typing import Dict, List, Optional, Tuple

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
_BANK_GEOM_PAIR_LIMIT = 256


class UnlearnFedAvgAggregator(ClientsAvgAggregator):
    """FedAvg variant with UNLEARN-style subspace discrimination."""
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model=model, device=device, config=config)
        self._device = torch.device(device) if not isinstance(
            device, torch.device) else device
        self._last_bases: Dict[str, torch.Tensor] = {}
        self._last_bases_per_client: Dict[int, Dict[str, torch.Tensor]] = {}
        self._ema_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def latest_bases(self) -> Dict[str, torch.Tensor]:
        """Return the latest Q bases to broadcast to clients."""
        return self._last_bases

    @property
    def latest_bases_per_client(self
                                ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return per-client bases payload when available."""
        return self._last_bases_per_client

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

        relevant_deltas = {key: client_deltas[key] for key in target_keys}
        bank_cfg = getattr(self.cfg.aggregator.unlearn, 'bank', None)
        bank_enabled = bool(getattr(bank_cfg, 'enable', False)) \
            if bank_cfg is not None else False
        send_flag = getattr(self.cfg.aggregator.unlearn,
                            'send_Q_to_clients', False)
        log_stats = bool(
            getattr(self.cfg.aggregator.unlearn, 'log_stats', True))
        round_idx = agg_info.get('round', 0)
        staleness = agg_info.get('staleness', [])
        client_ids = [client_id for client_id, _ in staleness]
        if len(client_ids) != len(client_states):
            client_ids = list(range(len(client_states)))

        updated_tensors: Dict[str, torch.Tensor] = {}

        if not bank_enabled:
            chunk_rows = self.cfg.aggregator.unlearn.chunk_rows
            proj_dtype = self.cfg.aggregator.unlearn.proj_dtype
            mode = self.cfg.aggregator.unlearn.mode.lower()
            lambda_shrink = self.cfg.aggregator.unlearn.lambda_shrink
            beta_shared = self.cfg.aggregator.unlearn.beta_shared
            alpha = self.cfg.aggregator.unlearn.alpha_global
            broadcast_cfg = getattr(self.cfg.aggregator.unlearn, 'broadcast',
                                    None)
            broadcast_kind = getattr(broadcast_cfg, 'kind', 'union').lower() \
                if broadcast_cfg else 'union'
            broadcast_ema_gamma = float(
                getattr(broadcast_cfg, 'ema_gamma', 0.0)) \
                if broadcast_cfg else 0.0
            broadcast_pack_dtype = getattr(broadcast_cfg, 'pack_dtype',
                                           proj_dtype) \
                if broadcast_cfg else proj_dtype
            broadcast_per_client = getattr(broadcast_cfg, 'per_client', True) \
                if broadcast_cfg is not None else False
            rank_cfg = getattr(self.cfg.aggregator.unlearn, 'rank', None)
            energy_target = getattr(rank_cfg, 'energy_target', 1.0) \
                if rank_cfg is not None else 1.0
            max_rank = getattr(rank_cfg, 'max_rank', 0) \
                if rank_cfg is not None else 0
            if isinstance(energy_target, dict):
                energy_default = float(
                    energy_target.get('default',
                                      energy_target.get('__default__', 1.0)))
            else:
                energy_default = float(energy_target)
            if isinstance(max_rank, dict):
                max_rank_default = int(
                    max_rank.get('default', max_rank.get('__default__', 0)))
            else:
                max_rank_default = int(max_rank)
            unique_parts, shared_parts, basis_map, stats_map = \
                self._discriminate(relevant_deltas, chunk_rows, proj_dtype,
                                   client_ids, self._ema_cache, energy_target,
                                   max_rank, broadcast_ema_gamma)
            self._update_ema_cache(client_ids, basis_map)
            stats = []

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
                    dtype=global_state[key].dtype,
                    device=global_state[key].device)
                updated_tensors[key] = updated_tensor

                key_stats = self._collect_stats(key, uniques, shareds,
                                                stats_map.get(key))
                stats.append(key_stats)

            per_client_payload = {}
            union_payload = {}
            should_broadcast = send_flag or broadcast_kind in {'loo', 'union'}
            if should_broadcast:
                if broadcast_kind == 'loo' and broadcast_per_client:
                    per_client_payload = self._build_per_client_payload(
                        basis_map, client_ids, proj_dtype, broadcast_pack_dtype,
                        round_idx, broadcast_kind)
                if not per_client_payload or broadcast_kind == 'union':
                    union_payload = self._build_union_payload(
                        relevant_deltas, proj_dtype, broadcast_pack_dtype,
                        round_idx,
                        broadcast_kind if broadcast_kind == 'union' else
                        'union', energy_default, max_rank_default)

            self._last_bases_per_client = per_client_payload
            self._last_bases = union_payload

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
                            log_payload[
                                f'{base_tag}/perp_parallel_ratio'] = ratio
                            log_payload[f'{base_tag}/rank_mean'] = record[
                                'rank_mean']
                            log_payload[f'{base_tag}/rank_full_mean'] = record[
                                'rank_full_mean']
                            log_payload[
                                f'{base_tag}/energy_mean'] = record[
                                    'energy_mean']
                        if log_payload:
                            wandb.log(log_payload, step=round_idx)
                    except ImportError:
                        logger.warning(
                            "cfg.wandb.use=True but wandb is not installed; "
                            "skip logging UNLEARN stats to wandb.")
                    except Exception as exc:
                        logger.warning(
                            "Failed to log UNLEARN stats to wandb: %s", exc)
                if log_stats:
                    for record in stats:
                        logger.info(
                            '[UNLEARN] key=%s ||perp||_F=%.4e ||parallel||_F='
                            '%.4e rank=%.1f/%.1f energy=%.3f mode=%s '
                            'alpha=%.3f lambda=%.3f beta=%.3f chunk=%d '
                            'dtype=%s device=%s', record['key'],
                            record['perp_norm'], record['parallel_norm'],
                            record['rank_mean'], record['rank_full_mean'],
                            record['energy_mean'], mode, alpha,
                            lambda_shrink, beta_shared, chunk_rows,
                            proj_dtype, self._device)
        else:
            self._last_bases = {}
            self._last_bases_per_client = {}
            proj_dtype = self.cfg.aggregator.unlearn.proj_dtype
            alpha = self.cfg.aggregator.unlearn.alpha_global
            beta_global = float(getattr(bank_cfg, 'beta_global', 1.0))
            beta_resid = float(getattr(bank_cfg, 'beta_resid', 0.0))
            r_global_cfg = int(getattr(bank_cfg, 'r_global', 0))
            r_client_cfg = int(getattr(bank_cfg, 'r_client', 0))
            energy_target = float(getattr(bank_cfg, 'energy_target', 0.0))
            log_geometry = bool(getattr(bank_cfg, 'log_geometry', False))
            bank_send_per_client = bool(
                getattr(bank_cfg, 'bank_send_per_client', False))
            bank_proj_mode = str(
                getattr(bank_cfg, 'bank_proj_mode', 'others_private'))
            include_shared = bank_proj_mode.lower() == 'others_plus_shared'
            bank_proj_rank_max = int(
                getattr(bank_cfg, 'bank_proj_rank_max', 0))
            work_dtype = self._resolve_proj_dtype(proj_dtype)
            bank_stats = []
            geom_stats = []
            shared_bases: Dict[str, torch.Tensor] = {}
            private_bases: Dict[str, Dict[int, torch.Tensor]] = {}

            with torch.no_grad():
                for key in target_keys:
                    base_tensor = cached_global.get(key)
                    if base_tensor is None:
                        continue
                    deltas_for_key = relevant_deltas.get(key, [])
                    if not deltas_for_key:
                        continue
                    deltas_fp = [
                        delta.to(device=self._device, dtype=work_dtype)
                        for delta in deltas_for_key
                    ]
                    stacked = torch.cat(deltas_fp, dim=0)
                    S_k = self._build_global_basis_for_bank(
                        stacked, r_global_cfg, energy_target, key)
                    if S_k is not None:
                        shared_bases[key] = S_k.detach()

                    delta_sum = torch.zeros_like(deltas_fp[0],
                                                 dtype=work_dtype,
                                                 device=self._device)
                    frac_records = []
                    priv_bases_local = []
                    for idx, delta_fp in enumerate(deltas_fp):
                        delta_fp = delta_fp.to(device=self._device,
                                               dtype=work_dtype)
                        d_glob, d_priv, d_res, priv_basis = \
                            self._decompose_with_bank(delta_fp, S_k,
                                                      r_client_cfg, key)
                        tilde = d_priv + beta_global * d_glob + \
                            beta_resid * d_res
                        weight = weights[idx] if idx < len(weights) else 0.0
                        delta_sum = delta_sum + weight * tilde

                        norm_sq = torch.sum(delta_fp * delta_fp).item()
                        denom = max(norm_sq, 1e-12)
                        frac_records.append({
                            'global':
                            torch.sum(d_glob * d_glob).item() / denom,
                            'private':
                            torch.sum(d_priv * d_priv).item() / denom,
                            'resid':
                            torch.sum(d_res * d_res).item() / denom,
                        })

                        client_id = client_ids[idx] if idx < len(client_ids) \
                            else idx
                        if priv_basis is not None and priv_basis.numel() > 0:
                            priv_bases_local.append(priv_basis)
                            priv_dict = private_bases.setdefault(key, {})
                            priv_dict[client_id] = priv_basis.detach()

                    if log_geometry:
                        geom_record = self._compute_bank_geometry(
                            key, S_k, priv_bases_local, work_dtype)
                        geom_record['numel'] = int(base_tensor.numel())
                        geom_stats.append(geom_record)

                    aggregated_delta = (alpha * delta_sum).to(
                        dtype=base_tensor.dtype)
                    updated_tensor = (base_tensor + aggregated_delta).to(
                        dtype=global_state[key].dtype,
                        device=global_state[key].device)
                    updated_tensors[key] = updated_tensor

                    if frac_records:
                        global_mean = sum(item['global']
                                          for item in frac_records) / \
                            len(frac_records)
                        private_mean = sum(item['private']
                                           for item in frac_records) / \
                            len(frac_records)
                        resid_mean = sum(item['resid']
                                         for item in frac_records) / \
                            len(frac_records)
                        bank_stats.append({
                            'key': key,
                            'global_fraction': global_mean,
                            'private_fraction': private_mean,
                            'resid_fraction': resid_mean,
                            'numel': int(base_tensor.numel()),
                        })

            if send_flag and bank_send_per_client and private_bases:
                per_client_payload = self._build_bank_per_client_payload(
                    private_bases, shared_bases, client_ids, proj_dtype,
                    bank_proj_rank_max, include_shared, round_idx)
            else:
                per_client_payload = {}
            self._last_bases_per_client = per_client_payload

            if bank_stats:
                total_numel = sum(item['numel'] for item in bank_stats)
                if total_numel > 0:
                    weighted_global = sum(
                        item['global_fraction'] * item['numel']
                        for item in bank_stats) / total_numel
                    weighted_private = sum(
                        item['private_fraction'] * item['numel']
                        for item in bank_stats) / total_numel
                    weighted_resid = sum(
                        item['resid_fraction'] * item['numel']
                        for item in bank_stats) / total_numel
                else:
                    weighted_global = weighted_private = weighted_resid = 0.0

                if getattr(self.cfg, 'wandb', None) and self.cfg.wandb.use:
                    try:
                        import wandb
                        round_idx = agg_info.get('round')
                        log_payload = {}
                        for record in bank_stats:
                            sanitized = record['key'].replace('.', '/')
                            base_tag = f'unlearn_bank/{sanitized}'
                            log_payload[
                                f'{base_tag}/global_fraction'] = record[
                                    'global_fraction']
                            log_payload[
                                f'{base_tag}/private_fraction'] = record[
                                    'private_fraction']
                            log_payload[f'{base_tag}/resid_fraction'] = record[
                                'resid_fraction']
                        log_payload['unlearn_bank/summary/global_fraction'] = \
                            weighted_global
                        log_payload['unlearn_bank/summary/private_fraction'] = \
                            weighted_private
                        log_payload['unlearn_bank/summary/resid_fraction'] = \
                            weighted_resid
                        if log_payload:
                            wandb.log(log_payload, step=round_idx)
                    except ImportError:
                        logger.warning(
                            "cfg.wandb.use=True but wandb is not installed; "
                            "skip logging bank stats to wandb.")
                    except Exception as exc:
                        logger.warning(
                            "Failed to log bank stats to wandb: %s", exc)
                if log_stats:
                    for record in bank_stats:
                        logger.info(
                            '[UNLEARN][bank] key=%s global_frac=%.4f '
                            'private_frac=%.4f resid_frac=%.4f '
                            'alpha=%.3f beta_global=%.3f beta_resid=%.3f '
                            'dtype=%s device=%s', record['key'],
                            record['global_fraction'],
                            record['private_fraction'],
                            record['resid_fraction'], alpha, beta_global,
                            beta_resid, proj_dtype, self._device)
                    logger.info(
                        '[UNLEARN][bank][avg] global_frac=%.4f '
                        'private_frac=%.4f resid_frac=%.4f '
                        'alpha=%.3f beta_global=%.3f beta_resid=%.3f',
                        weighted_global, weighted_private, weighted_resid,
                        alpha, beta_global, beta_resid)
                if log_geometry and geom_stats:
                    total_numel_geom = sum(item['numel']
                                           for item in geom_stats)
                    if total_numel_geom > 0:
                        weighted_orth = sum(
                            item['orth_mean'] * item['numel']
                            for item in geom_stats) / total_numel_geom
                        weighted_cross = sum(
                            item['cross_mean'] * item['numel']
                            for item in geom_stats) / total_numel_geom
                    else:
                        weighted_orth = 0.0
                        weighted_cross = 0.0

                    if getattr(self.cfg, 'wandb', None) and self.cfg.wandb.use:
                        try:
                            import wandb
                            round_idx = agg_info.get('round')
                            log_payload = {}
                            for record in geom_stats:
                                sanitized = record['key'].replace('.', '/')
                                base_tag = f'unlearn_bank_geom/{sanitized}'
                                log_payload[
                                    f'{base_tag}/orth_mean'] = record[
                                        'orth_mean']
                                log_payload[
                                    f'{base_tag}/orth_max'] = record[
                                        'orth_max']
                                log_payload[
                                    f'{base_tag}/cross_mean'] = record[
                                        'cross_mean']
                                log_payload[
                                    f'{base_tag}/cross_max'] = record[
                                        'cross_max']
                                log_payload[
                                    f'{base_tag}/num_pairs'] = record[
                                        'num_pairs']
                            log_payload[
                                'unlearn_bank_geom/summary/orth_mean'] = \
                                weighted_orth
                            log_payload[
                                'unlearn_bank_geom/summary/cross_mean'] = \
                                weighted_cross
                            if log_payload:
                                wandb.log(log_payload, step=round_idx)
                        except ImportError:
                            logger.warning(
                                "cfg.wandb.use=True but wandb is not "
                                "installed; skip logging geometry stats to "
                                "wandb.")
                        except Exception as exc:
                            logger.warning(
                                "Failed to log geometry stats to wandb: %s",
                                exc)

                    if log_stats:
                        for record in geom_stats:
                            logger.info(
                                '[UNLEARN][bank][geom] key=%s orth_mean=%.4f '
                                'orth_max=%.4f cross_mean=%.4f '
                                'cross_max=%.4f pairs=%d failed=%s',
                                record['key'], record['orth_mean'],
                                record['orth_max'], record['cross_mean'],
                                record['cross_max'], record['num_pairs'],
                                record['failed'])
                        logger.info(
                            '[UNLEARN][bank][geom][avg] orth_mean=%.4f '
                            'cross_mean=%.4f',
                            weighted_orth, weighted_cross)

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
        client_ids: List[int],
        prev_bases: Dict[int, Dict[str, torch.Tensor]],
        energy_target,
        max_rank,
        ema_gamma: float,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]],
               Dict[str, List[Optional[torch.Tensor]]],
               Dict[str, List[Dict[str, float]]]]:
        return chunked_discrimination(deltas,
                                      chunk_rows,
                                      proj_dtype,
                                      client_ids=client_ids,
                                      prev_bases=prev_bases,
                                      energy_target=energy_target,
                                      max_rank=max_rank,
                                      ema_gamma=ema_gamma)

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

    def _update_ema_cache(self, client_ids: List[int],
                          basis_map: Dict[str, List[Optional[torch.Tensor]]]):
        if not client_ids:
            return
        for idx, client_id in enumerate(client_ids):
            cache = self._ema_cache.setdefault(client_id, {})
            for key, basis_list in basis_map.items():
                if idx >= len(basis_list):
                    continue
                basis = basis_list[idx]
                if basis is None:
                    cache.pop(key, None)
                else:
                    cache[key] = basis.detach().to(device='cpu',
                                                   dtype=torch.float32)

    def _build_per_client_payload(
        self,
        basis_map: Dict[str, List[Optional[torch.Tensor]]],
        client_ids: List[int],
        proj_dtype: str,
        pack_dtype: str,
        round_idx: int,
        kind: str,
    ) -> Dict[int, Dict[str, object]]:
        if not client_ids:
            return {}
        pack_torch_dtype = self._resolve_proj_dtype(pack_dtype)
        payload: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx, client_id in enumerate(client_ids):
            q_dict = {}
            for key, basis_list in basis_map.items():
                if idx >= len(basis_list):
                    continue
                basis = basis_list[idx]
                if basis is None:
                    continue
                packed = self._pack_basis_tensor(basis, pack_torch_dtype)
                q_dict[key] = packed
            if q_dict:
                payload[client_id] = {
                    'kind': kind,
                    'round': round_idx,
                    'proj_dtype': proj_dtype,
                    'pack_dtype': pack_dtype,
                    'Q': q_dict
                }
        return payload

    def _build_union_payload(
        self,
        deltas: Dict[str, List[torch.Tensor]],
        proj_dtype: str,
        pack_dtype: str,
        round_idx: int,
        kind: str,
        energy_target,
        max_rank,
    ) -> Dict[str, object]:
        proj_torch_dtype = self._resolve_proj_dtype(proj_dtype)
        pack_torch_dtype = self._resolve_proj_dtype(pack_dtype)
        q_dict = {}
        for key, tensors in deltas.items():
            if not tensors:
                continue
            stacked = torch.cat([
                delta.to(self._device) for delta in tensors
            ],
                                 dim=0).to(dtype=proj_torch_dtype)
            basis = qr_basis_from_concat(stacked,
                                         proj_torch_dtype,
                                         energy_target=energy_target,
                                         max_rank=max_rank)
            if basis is None:
                continue
            q_dict[key] = self._pack_basis_tensor(basis, pack_torch_dtype)
        if not q_dict:
            return {}
        return {
            'kind': kind,
            'round': round_idx,
            'proj_dtype': proj_dtype,
            'pack_dtype': pack_dtype,
            'Q': q_dict
        }

    def _pack_basis_tensor(self, tensor: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
        return tensor.detach().to(device='cpu', dtype=dtype)

    def _collect_stats(self,
                       key: str,
                       uniques: List[torch.Tensor],
                       shareds: List[torch.Tensor],
                       basis_stats: Optional[List[Dict[str, float]]] = None
                       ) -> Dict[str, float]:
        unique_norms = [
            torch.linalg.norm(tensor.to(device='cpu', dtype=torch.float32))
            for tensor in uniques
        ]
        shared_norms = [
            torch.linalg.norm(tensor.to(device='cpu', dtype=torch.float32))
            for tensor in shareds
        ]
        if basis_stats:
            ranks = torch.tensor([stat.get('rank', 0) for stat in basis_stats],
                                 dtype=torch.float32)
            rank_full = torch.tensor(
                [stat.get('full_rank', 0) for stat in basis_stats],
                dtype=torch.float32)
            energies = torch.tensor(
                [stat.get('energy', 0.0) for stat in basis_stats],
                dtype=torch.float32)
            rank_mean = ranks.mean().item()
            rank_full_mean = rank_full.mean().item()
            energy_mean = energies.mean().item()
        else:
            rank_mean = 0.0
            rank_full_mean = 0.0
            energy_mean = 0.0
        return {
            'key': key,
            'perp_norm': torch.stack(unique_norms).mean().item(),
            'parallel_norm': torch.stack(shared_norms).mean().item(),
            'rank_mean': rank_mean,
            'rank_full_mean': rank_full_mean,
            'energy_mean': energy_mean
        }

    def _build_bank_per_client_payload(
        self,
        private_bases: Dict[str, Dict[int, torch.Tensor]],
        shared_bases: Dict[str, torch.Tensor],
        client_ids: List[int],
        proj_dtype: str,
        rank_max: int,
        include_shared: bool,
        round_idx: int,
    ) -> Dict[int, Dict[str, object]]:
        if not private_bases:
            return {}
        dtype = self._resolve_proj_dtype(proj_dtype)
        unique_clients = list(dict.fromkeys(client_ids)) if client_ids else \
            sorted({
            cid
            for basis_map in private_bases.values() for cid in basis_map
        })
        payload: Dict[int, Dict[str, object]] = {}
        for client_id in unique_clients:
            q_dict: Dict[str, torch.Tensor] = {}
            for key, basis_map in private_bases.items():
                others = []
                for other_id, basis in basis_map.items():
                    if other_id == client_id:
                        continue
                    others.append(basis.to(device=self._device,
                                           dtype=dtype))
                if include_shared:
                    shared = shared_bases.get(key)
                    if shared is not None:
                        others.append(
                            shared.to(device=self._device, dtype=dtype))
                if not others:
                    continue
                concat = torch.cat(others, dim=1)
                ortho = self._orthonormalize_basis(concat, rank_max, dtype)
                if ortho is None or ortho.numel() == 0:
                    continue
                q_dict[key] = self._pack_basis_tensor(ortho, dtype)
            if q_dict:
                payload[client_id] = {
                    'kind': 'bank_per_client',
                    'round': round_idx,
                    'proj_dtype': proj_dtype,
                    'Q': q_dict
                }
        return payload

    def _orthonormalize_basis(self,
                              tensor: torch.Tensor,
                              rank_max: int,
                              proj_dtype: torch.dtype) -> Optional[torch.Tensor]:
        if tensor is None or tensor.numel() == 0:
            return None
        matrix = tensor.to(device=self._device, dtype=proj_dtype)
        try:
            q, _ = torch.linalg.qr(matrix, mode='reduced')
        except RuntimeError as exc:
            logger.warning(
                '[UNLEARN][bank] QR failed when building per-client bases: %s',
                exc)
            return None
        if rank_max > 0:
            cols = min(rank_max, q.shape[1])
            q = q[:, :cols]
        return q

    def _build_global_basis_for_bank(self, stacked: torch.Tensor,
                                     r_global_cfg: int,
                                     energy_target: float,
                                     key: str) -> Optional[torch.Tensor]:
        if stacked.numel() == 0:
            return None
        try:
            _, singular_vals, v_h = torch.linalg.svd(stacked,
                                                     full_matrices=False)
        except RuntimeError as exc:
            logger.warning('[UNLEARN][bank] failed SVD for key %s: %s', key,
                           exc)
            return None
        if v_h.numel() == 0:
            return None
        basis = v_h.transpose(-1, -2)
        num_cols = basis.shape[1]
        target_rank = max(r_global_cfg, 0)
        if energy_target > 0.0 and singular_vals.numel() > 0:
            sing_sq = singular_vals * singular_vals
            total = torch.sum(sing_sq)
            if total.item() > 0:
                cumulative = torch.cumsum(sing_sq, dim=-1) / total
                meet = (cumulative >= energy_target).nonzero(
                    as_tuple=False)
                if meet.numel() > 0:
                    target_rank = int(meet[0].item()) + 1
                else:
                    target_rank = num_cols
        target_rank = min(target_rank, num_cols)
        if target_rank <= 0:
            return None
        return basis[:, :target_rank]

    def _decompose_with_bank(self,
                             delta: torch.Tensor,
                             global_basis: Optional[torch.Tensor],
                             r_client_cfg: int,
                             key: str) -> Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor,
                                                Optional[torch.Tensor]]:
        if global_basis is not None:
            d_glob = (delta @ global_basis) @ global_basis.transpose(-1, -2)
        else:
            d_glob = torch.zeros_like(delta)
        residual = delta - d_glob
        if r_client_cfg <= 0:
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        delta_norm = torch.linalg.norm(delta).item()
        residual_norm = torch.linalg.norm(residual).item()
        threshold = 1e-8 * max(delta_norm, 1e-12)
        if residual_norm <= threshold:
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        try:
            _, _, v_h = torch.linalg.svd(residual, full_matrices=False)
        except RuntimeError as exc:
            logger.warning(
                '[UNLEARN][bank] failed private SVD for key %s: %s', key, exc)
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        if v_h.numel() == 0:
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        basis = v_h.transpose(-1, -2)
        r_client = min(max(r_client_cfg, 0), basis.shape[1])
        if r_client == 0:
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        priv_basis = basis[:, :r_client]
        if global_basis is not None:
            proj = global_basis @ (
                global_basis.transpose(-1, -2) @ priv_basis)
            priv_basis = priv_basis - proj
            if priv_basis.numel() == 0:
                priv_basis = None
            else:
                try:
                    q, _ = torch.linalg.qr(priv_basis, mode='reduced')
                except RuntimeError as exc:
                    logger.warning(
                        '[UNLEARN][bank] QR failed for key %s: %s', key, exc)
                    q = None
                if q is None or q.shape[1] == 0:
                    priv_basis = None
                else:
                    cols = min(r_client, q.shape[1])
                    priv_basis = q[:, :cols]
        if priv_basis is None or priv_basis.shape[1] == 0:
            d_priv = torch.zeros_like(delta)
            return d_glob, d_priv, residual, None
        d_priv = (residual @ priv_basis) @ priv_basis.transpose(-1, -2)
        d_res = delta - d_glob - d_priv
        return d_glob, d_priv, d_res, priv_basis

    def _compute_bank_geometry(self, key: str,
                               global_basis: Optional[torch.Tensor],
                               priv_bases: List[torch.Tensor],
                               proj_dtype: torch.dtype) -> Dict[str, float]:
        record = {
            'key': key,
            'orth_mean': 0.0,
            'orth_max': 0.0,
            'cross_mean': 0.0,
            'cross_max': 0.0,
            'num_pairs': 0,
            'failed': False,
        }
        if not priv_bases:
            return record
        try:
            orth_vals = []
            if global_basis is not None and global_basis.numel() > 0:
                S = global_basis.to(device=self._device, dtype=proj_dtype)
                for pb in priv_bases:
                    if pb is None or pb.numel() == 0:
                        continue
                    inner = S.transpose(-1, -2) @ pb
                    denom = (S.shape[1] * pb.shape[1])**0.5
                    val = torch.linalg.norm(inner).item()
                    if denom > 0:
                        val = val / denom
                    orth_vals.append(val)
            if orth_vals:
                record['orth_mean'] = sum(orth_vals) / len(orth_vals)
                record['orth_max'] = max(orth_vals)

            if len(priv_bases) >= 2:
                cross_vals = []
                pairs = 0
                for i in range(len(priv_bases)):
                    pb_i = priv_bases[i]
                    if pb_i is None or pb_i.numel() == 0:
                        continue
                    for j in range(i + 1, len(priv_bases)):
                        pb_j = priv_bases[j]
                        if pb_j is None or pb_j.numel() == 0:
                            continue
                        inner = pb_i.transpose(-1, -2) @ pb_j
                        denom = (pb_i.shape[1] * pb_j.shape[1])**0.5
                        val = torch.linalg.norm(inner).item()
                        if denom > 0:
                            val = val / denom
                        cross_vals.append(val)
                        pairs += 1
                        if pairs >= _BANK_GEOM_PAIR_LIMIT:
                            break
                    if pairs >= _BANK_GEOM_PAIR_LIMIT:
                        break
                record['num_pairs'] = pairs
                if cross_vals:
                    record['cross_mean'] = sum(cross_vals) / len(cross_vals)
                    record['cross_max'] = max(cross_vals)
        except Exception as exc:
            record['failed'] = True
            record['orth_mean'] = 0.0
            record['orth_max'] = 0.0
            record['cross_mean'] = 0.0
            record['cross_max'] = 0.0
            record['num_pairs'] = 0
            logger.warning(
                '[UNLEARN][bank] geometry stats failed for key %s: %s', key,
                exc)
        return record

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
            'bfloat16': torch.bfloat16,
        }
        return mapping.get(name.lower(), torch.float32)
