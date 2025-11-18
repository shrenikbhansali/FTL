from collections.abc import Mapping
import torch
from typing import Dict, Iterable, List, Optional, Tuple


def select_target_params(
    state_dict: Dict[str, torch.Tensor],
    only_lora: bool,
    target_modules: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Select parameters to operate on based on name heuristics."""
    selected = {}
    module_terms = tuple(target_modules)
    for name, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue
        if only_lora:
            if 'lora_A' in name or 'lora_B' in name:
                selected[name] = tensor
        else:
            if any(term in name for term in module_terms):
                selected[name] = tensor
    return selected


def compute_client_deltas(
    client_states: List[Dict[str, torch.Tensor]],
    server_state: Dict[str, torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """Return deltas between client parameters and the reference server."""
    deltas = []
    with torch.no_grad():
        for client_state in client_states:
            local_delta = {}
            for name, tensor in client_state.items():
                if name not in server_state:
                    continue
                local_delta[name] = tensor - server_state[name]
            deltas.append(local_delta)
    return deltas


def _resolve_proj_dtype(name: str) -> torch.dtype:
    if name.lower() in ['float32', 'fp32']:
        return torch.float32
    if name.lower() in ['float64', 'fp64', 'double']:
        return torch.float64
    if name.lower() in ['float16', 'fp16', 'half']:
        return torch.float16
    return torch.float32


def _resolve_named_setting(setting, key: str, default_value):
    """Resolve optional mapping-based overrides."""
    if isinstance(setting, Mapping):
        if key in setting:
            return setting[key]
        for fallback in ['default', '__default__']:
            if fallback in setting:
                return setting[fallback]
        return default_value
    return setting


def ema_blend(prev: Optional[torch.Tensor],
              current: torch.Tensor,
              gamma: float,
              eps: float = 1e-6) -> torch.Tensor:
    """Blend two bases and re-orthonormalize with QR."""
    if current is None:
        return prev
    if prev is None or gamma <= eps:
        return current
    if gamma >= 1.0 - eps:
        return prev
    target_cols = max(prev.shape[1], current.shape[1])
    if target_cols == 0:
        return current

    def _pad_columns(tensor: torch.Tensor, cols: int) -> torch.Tensor:
        if tensor.shape[1] == cols:
            return tensor
        pad_cols = cols - tensor.shape[1]
        pad = torch.zeros(tensor.shape[0],
                          pad_cols,
                          device=tensor.device,
                          dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)

    prev_pad = _pad_columns(prev, target_cols)
    curr_pad = _pad_columns(current, target_cols)
    blended = gamma * prev_pad + (1.0 - gamma) * curr_pad
    if torch.linalg.norm(blended) <= eps:
        return current
    try:
        q, _ = torch.linalg.qr(blended, mode='reduced')
    except RuntimeError:
        blended = blended + eps * torch.randn_like(blended)
        q, _ = torch.linalg.qr(blended, mode='reduced')
    return q


def qr_basis_from_concat(matrix: torch.Tensor,
                         proj_dtype: torch.dtype,
                         eps: float = 1e-6,
                         energy_target: float = 1.0,
                         max_rank: int = 0,
                         return_info: bool = False):
    """Return (optionally truncated) orthonormal basis Q for the row space."""
    if matrix.numel() == 0 or matrix.shape[0] == 0:
        return (None, None) if return_info else None
    work = matrix.to(dtype=proj_dtype)
    if not torch.isfinite(work).all():
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.linalg.norm(work) <= eps:
        return (None, None) if return_info else None
    try:
        _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    except (RuntimeError, torch.linalg.LinAlgError):
        work = work + eps * torch.randn_like(work)
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    mask = singular_values > eps
    if mask.sum() == 0:
        return (None, None) if return_info else None

    squared = singular_values**2
    total_energy = squared.sum()
    if total_energy <= eps:
        return (None, None) if return_info else None
    cumulative = torch.cumsum(squared, dim=0)
    target_energy = float(max(min(energy_target, 1.0), 0.0))
    if target_energy <= 0.0:
        required_rank = int(mask.sum().item())
    else:
        threshold = target_energy * total_energy
        idx = torch.searchsorted(cumulative, torch.tensor(threshold,
                                                          device=cumulative.device))
        required_rank = int(idx.item()) + 1
    available_rank = int(mask.sum().item())
    if max_rank and max_rank > 0:
        required_rank = min(required_rank, max_rank)
    required_rank = max(0, min(required_rank, available_rank))
    if required_rank == 0:
        return (None, None) if return_info else None
    energy_ratio = cumulative[required_rank - 1] / total_energy
    basis = vh[mask][:required_rank].transpose(0, 1).contiguous()
    info = {
        'rank': required_rank,
        'full_rank': available_rank,
        'energy': float(energy_ratio.item())
    }
    if return_info:
        return basis, info
    return basis


def project_rows(matrix: torch.Tensor,
                 Q: Optional[torch.Tensor]) -> torch.Tensor:
    """Project matrix onto the span of Q."""
    if Q is None:
        return torch.zeros_like(matrix)
    return (matrix @ Q) @ Q.transpose(0, 1)


def discriminate_one_key(
    deltas: List[torch.Tensor],
    chunk_rows: int,
    proj_dtype: torch.dtype,
    energy_target: float,
    max_rank: int,
    ema_gamma: float,
    prev_bases: Optional[List[Optional[torch.Tensor]]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]],
           List[Dict[str, float]]]:
    """Split each delta into parallel and orthogonal components."""
    if not deltas:
        return [], [], [], []
    num_clients = len(deltas)
    device = deltas[0].device
    perp_outputs: List[torch.Tensor] = []
    parallel_outputs: List[torch.Tensor] = []
    bases_outputs: List[Optional[torch.Tensor]] = []
    stats_outputs: List[Dict[str, float]] = []

    with torch.no_grad():
        # Pre-compute union bases for each client
        bases: List[Optional[torch.Tensor]] = []
        prev_bases = prev_bases or [None] * num_clients
        for idx in range(num_clients):
            if num_clients == 1:
                bases.append(None)
                continue
            others = [
                deltas[j].to(device=device, dtype=proj_dtype)
                for j in range(num_clients) if j != idx
            ]
            if not others:
                bases.append(None)
                continue
            stacked = torch.cat(others, dim=0)
            Q, info = qr_basis_from_concat(stacked,
                                           proj_dtype,
                                           energy_target=energy_target,
                                           max_rank=max_rank,
                                           return_info=True)
            if Q is not None and ema_gamma > 0.0:
                prev = prev_bases[idx]
                if prev is not None:
                    prev = prev.to(device=device, dtype=proj_dtype)
                Q = ema_blend(prev, Q, ema_gamma)
            bases.append(Q)
            stats_outputs.append({
                'rank': 0 if info is None else info.get('rank', 0),
                'full_rank': 0 if info is None else info.get('full_rank', 0),
                'energy': 0.0 if info is None else info.get('energy', 0.0)
            })
        if len(stats_outputs) != num_clients:
            stats_outputs = [{
                'rank': bases[idx].shape[1] if bases[idx] is not None else 0,
                'full_rank': bases[idx].shape[1]
                if bases[idx] is not None else 0,
                'energy': 0.0
            } for idx in range(num_clients)]

        for idx, delta in enumerate(deltas):
            Q = bases[idx]
            current = delta.to(device=device, dtype=proj_dtype)
            if Q is None:
                parallel = torch.zeros_like(current)
                orthogonal = current
            else:
                parallel = project_rows(current, Q)
                orthogonal = current - parallel
            parallel_outputs.append(parallel.to(dtype=delta.dtype))
            perp_outputs.append(orthogonal.to(dtype=delta.dtype))
            bases_outputs.append(Q)
            stats_outputs[idx] = {
                'rank': 0 if Q is None else Q.shape[1],
                'full_rank': stats_outputs[idx].get('full_rank', 0),
                'energy': stats_outputs[idx].get('energy', 0.0)
            }

    return perp_outputs, parallel_outputs, bases_outputs, stats_outputs


def chunked_discrimination(
    deltas_by_client: Dict[str, List[torch.Tensor]],
    chunk_rows: int,
    proj_dtype: str,
    client_ids: Optional[List[int]] = None,
    prev_bases: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    energy_target: float = 1.0,
    max_rank: int = 0,
    ema_gamma: float = 0.0,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]],
           Dict[str, List[Optional[torch.Tensor]]],
           Dict[str, List[Dict[str, float]]]]:
    """Apply discrimination across all parameter keys."""
    dtype = _resolve_proj_dtype(proj_dtype)
    unique_parts, shared_parts = {}, {}
    bases_map: Dict[str, List[Optional[torch.Tensor]]] = {}
    stats_map: Dict[str, List[Dict[str, float]]] = {}
    num_clients = len(next(iter(deltas_by_client.values()))) \
        if deltas_by_client else 0
    fallback_prev = [None] * num_clients
    if isinstance(energy_target, Mapping):
        default_energy = float(
            energy_target.get('default',
                              energy_target.get('__default__', 1.0)))
    else:
        default_energy = float(energy_target)
    if isinstance(max_rank, Mapping):
        default_rank = int(max_rank.get('default',
                                        max_rank.get('__default__', 0)))
    else:
        default_rank = int(max_rank)

    for name, tensors in deltas_by_client.items():
        if client_ids and prev_bases:
            prev_list = [
                prev_bases.get(client_id, {}).get(name)
                for client_id in client_ids
            ]
        else:
            prev_list = fallback_prev
        target_energy = float(
            _resolve_named_setting(energy_target, name, default_energy))
        rank_cap = int(
            _resolve_named_setting(max_rank, name, default_rank))
        unique, shared, bases, stats = discriminate_one_key(
            tensors, chunk_rows, dtype, target_energy, rank_cap, ema_gamma,
            prev_list)
        unique_parts[name] = unique
        shared_parts[name] = shared
        bases_map[name] = bases
        stats_map[name] = stats
    return unique_parts, shared_parts, bases_map, stats_map


def maybe_clear_cuda(device: torch.device):
    """Clear CUDA cache when tensors live on GPU."""
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.empty_cache()
