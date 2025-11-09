import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_aggregator_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # aggregator related options
    # fs has supported the robust aggregation rules of 'krum', 'median',
    # 'trimmedmean', 'bulyan' and 'normbounding', the use case is refered
    #  to tests/test_robust_aggregators.py
    # ---------------------------------------------------------------------- #
    cfg.aggregator = CN()
    cfg.aggregator.type = ''
    cfg.aggregator.robust_rule = 'fedavg'
    cfg.aggregator.byzantine_node_num = 0
    cfg.aggregator.BFT_args = CN(new_allowed=True)

    # For ATC method
    cfg.aggregator.num_agg_groups = 1
    cfg.aggregator.num_agg_topk = []
    cfg.aggregator.inside_weight = 1.0
    cfg.aggregator.outside_weight = 0.0

    # UNLEARN aggregation controls
    cfg.aggregator.unlearn = CN()
    cfg.aggregator.unlearn.enable = False
    cfg.aggregator.unlearn.mode = 'shrink'  # {'shrink', 'mean_shared'}
    cfg.aggregator.unlearn.lambda_shrink = 1.0
    cfg.aggregator.unlearn.beta_shared = 0.2
    cfg.aggregator.unlearn.alpha_global = 1.0
    cfg.aggregator.unlearn.only_lora = True
    cfg.aggregator.unlearn.target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj'
    ]
    cfg.aggregator.unlearn.proj_dtype = 'float32'
    cfg.aggregator.unlearn.chunk_rows = 4096
    cfg.aggregator.unlearn.send_Q_to_clients = False

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_aggregator_cfg)


def assert_aggregator_cfg(cfg):

    if cfg.aggregator.byzantine_node_num == 0 and \
            cfg.aggregator.robust_rule in \
            ['krum', 'normbounding', 'median', 'trimmedmean', 'bulyan']:
        logging.warning(
            f'Although {cfg.aggregator.robust_rule} aggregtion rule is '
            'applied, we found that cfg.aggregator.byzantine_node_num == 0')


register_config('aggregator', extend_aggregator_cfg)
