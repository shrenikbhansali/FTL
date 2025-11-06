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


def qr_basis_from_concat(matrix: torch.Tensor,
                         proj_dtype: torch.dtype,
                         eps: float = 1e-6) -> Optional[torch.Tensor]:
    """Return orthonormal basis Q for the row space of matrix."""
    if matrix.numel() == 0 or matrix.shape[0] == 0:
        return None
    work = matrix.to(dtype=proj_dtype)
    if not torch.isfinite(work).all():
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.linalg.norm(work) <= eps:
        return None
    try:
        _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    except (RuntimeError, torch.linalg.LinAlgError):
        work = work + eps * torch.randn_like(work)
        work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    mask = singular_values > eps
    if mask.sum() == 0:
        return None
    return vh[mask].transpose(0, 1).contiguous()


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
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Split each delta into parallel and orthogonal components."""
    if not deltas:
        return [], []
    num_clients = len(deltas)
    device = deltas[0].device
    perp_outputs: List[torch.Tensor] = []
    parallel_outputs: List[torch.Tensor] = []

    with torch.no_grad():
        # Pre-compute union bases for each client
        bases: List[Optional[torch.Tensor]] = []
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
            Q = qr_basis_from_concat(stacked, proj_dtype)
            bases.append(Q)

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

    return perp_outputs, parallel_outputs


def chunked_discrimination(
    deltas_by_client: Dict[str, List[torch.Tensor]],
    chunk_rows: int,
    proj_dtype: str,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """Apply discrimination across all parameter keys."""
    dtype = _resolve_proj_dtype(proj_dtype)
    unique_parts, shared_parts = {}, {}
    for name, tensors in deltas_by_client.items():
        unique, shared = discriminate_one_key(tensors, chunk_rows, dtype)
        unique_parts[name] = unique
        shared_parts[name] = shared
    return unique_parts, shared_parts


def maybe_clear_cuda(device: torch.device):
    """Clear CUDA cache when tensors live on GPU."""
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.empty_cache()
