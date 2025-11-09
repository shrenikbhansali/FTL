import torch

from federatedscope.llm.algo import (discriminate_one_key, project_rows,
                                     qr_basis_from_concat)
from federatedscope.core.aggregators.unlearn_fedavg_aggregator import \
    UnlearnFedAvgAggregator
from federatedscope.core.configs.config import global_cfg


def _build_deltas():
    base = torch.tensor([[1.0, 0.5], [0.0, 0.0]], dtype=torch.float32)
    delta_1 = base.clone()
    delta_2 = torch.tensor([[0.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
    delta_3 = torch.tensor([[0.5, -0.5], [0.25, 0.0]], dtype=torch.float32)
    return [delta_1, delta_2, delta_3]


def test_perpendicular_component_is_orthogonal_to_rowspace():
    deltas = _build_deltas()
    unique, _ = discriminate_one_key(deltas,
                                     chunk_rows=1,
                                     proj_dtype=torch.float32)

    for idx, perp in enumerate(unique):
        others = [deltas[j] for j in range(len(deltas)) if j != idx]
        stacked = torch.cat(others, dim=0)
        Q = qr_basis_from_concat(stacked, torch.float32)
        if Q is None:
            continue
        projection = project_rows(perp.to(dtype=torch.float32), Q)
        assert torch.allclose(projection,
                              torch.zeros_like(projection),
                              atol=1e-6)


def test_identical_clients_have_zero_unique_component():
    ident = torch.ones((2, 2), dtype=torch.float32)
    deltas = [ident.clone() for _ in range(3)]
    unique, shared = discriminate_one_key(deltas,
                                          chunk_rows=4,
                                          proj_dtype=torch.float32)

    for perp, parallel in zip(unique, shared):
        assert torch.allclose(perp, torch.zeros_like(perp), atol=1e-6)
        assert torch.allclose(parallel, ident, atol=1e-6)


def test_unlearn_aggregator_matches_fedavg_on_orthogonal_clients():
    cfg = global_cfg.clone()
    cfg.defrost()
    cfg.aggregator.unlearn.enable = True
    cfg.aggregator.unlearn.only_lora = False
    cfg.aggregator.unlearn.target_modules = ['weight']
    cfg.aggregator.unlearn.send_Q_to_clients = True
    cfg.aggregator.unlearn.mode = 'shrink'
    cfg.aggregator.unlearn.lambda_shrink = 1.0
    cfg.aggregator.unlearn.beta_shared = 0.2
    cfg.aggregator.unlearn.alpha_global = 1.0
    cfg.freeze()

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    aggregator = UnlearnFedAvgAggregator(model=model, device='cpu', config=cfg)

    client_a = {'weight': torch.tensor([[1.0, 0.0], [0.0, 0.0]])}
    client_b = {'weight': torch.tensor([[0.0, 0.0], [0.0, 1.0]])}
    agg_info = {
        'client_feedback': [
            (1, {k: v.clone()
                 for k, v in client_a.items()}),
            (1, {k: v.clone()
                 for k, v in client_b.items()}),
        ]
    }

    result = aggregator.aggregate(agg_info)
    expected = (client_a['weight'] + client_b['weight']) / 2.0

    assert 'weight' in result
    assert torch.allclose(result['weight'], expected, atol=1e-6)
    assert aggregator.latest_bases
    basis = aggregator.latest_bases.get('weight')
    assert basis is not None
    assert basis.ndim == 2
