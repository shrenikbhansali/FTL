import os

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_evaluation_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # Evaluation related options
    # ---------------------------------------------------------------------- #
    cfg.eval = CN(
        new_allowed=True)  # allow user to add their settings under `cfg.eval`

    cfg.eval.freq = 1
    cfg.eval.metrics = []
    cfg.eval.split = ['test', 'val']
    cfg.eval.report = ['weighted_avg', 'avg', 'fairness',
                       'raw']  # by default, we report comprehensive results
    cfg.eval.best_res_update_round_wise_key = "val_loss"

    # Monitoring, e.g., 'dissim' for B-local dissimilarity
    cfg.eval.monitoring = []
    cfg.eval.count_flops = True

    # ---------------------------------------------------------------------- #
    # wandb related options
    # ---------------------------------------------------------------------- #
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.name_user = ''
    cfg.wandb.name_project = ''
    cfg.wandb.online_track = True
    cfg.wandb.client_train_info = False

    # Allow environment variables to auto-enable wandb for legacy configs.
    env_use = os.getenv("WANDB_USE")
    env_project = os.getenv("WANDB_PROJECT")
    env_entity = os.getenv("WANDB_ENTITY") or os.getenv("WANDB_NAME")
    env_online = os.getenv("WANDB_ONLINE")
    env_client_info = os.getenv("WANDB_CLIENT_TRAIN_INFO")

    def _truthy(val):
        return str(val).lower() in ["1", "true", "yes", "on"]

    if env_use is not None:
        cfg.wandb.use = _truthy(env_use)
    elif env_project and env_entity and os.getenv("WANDB_API_KEY"):
        # Default to enable when all credentials are present.
        cfg.wandb.use = True

    if env_entity:
        cfg.wandb.name_user = env_entity
    if env_project:
        cfg.wandb.name_project = env_project
    if env_online is not None:
        cfg.wandb.online_track = _truthy(env_online)
    if env_client_info is not None:
        cfg.wandb.client_train_info = _truthy(env_client_info)

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_evaluation_cfg)


def assert_evaluation_cfg(cfg):
    pass


register_config("eval", extend_evaluation_cfg)
