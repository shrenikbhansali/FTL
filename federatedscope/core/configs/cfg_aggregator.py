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
    cfg.aggregator.unlearn.chunk_rows = 0
    cfg.aggregator.unlearn.send_Q_to_clients = False
    cfg.aggregator.unlearn.broadcast = CN()
    cfg.aggregator.unlearn.broadcast.kind = 'loo'  # {'loo', 'union'}
    cfg.aggregator.unlearn.broadcast.ema_gamma = 0.0
    cfg.aggregator.unlearn.broadcast.pack_dtype = 'float16'
    cfg.aggregator.unlearn.broadcast.per_client = True
    cfg.aggregator.unlearn.rank = CN()
    cfg.aggregator.unlearn.rank.energy_target = 0.9
    cfg.aggregator.unlearn.rank.max_rank = 0  # 0 denotes unlimited
    cfg.aggregator.unlearn.bank = CN()
    cfg.aggregator.unlearn.bank.enable = False
    cfg.aggregator.unlearn.bank.r_global = 4
    cfg.aggregator.unlearn.bank.r_client = 4
    cfg.aggregator.unlearn.bank.beta_global = 1.0
    cfg.aggregator.unlearn.bank.beta_resid = 0.0
    cfg.aggregator.unlearn.bank.energy_target = 0.0
    cfg.aggregator.unlearn.bank.bank_send_per_client = False
    cfg.aggregator.unlearn.bank.bank_proj_mode = \
        'others_private'  # {'others_private', 'others_plus_shared'}
    cfg.aggregator.unlearn.bank.bank_proj_rank_max = 0

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
