import argparse
import json
import os
import subprocess
import sys


COMMON_KEYS = {
    "sample_client_rate": "federate.sample_client_rate",
    "local_update_steps": "train.local_update_steps",
}

BASIC_SHARD_KEYS = {
    "client_shards_enable": "data.tulu3_federated.sharding.enable",
    "client_shards_default": "data.tulu3_federated.sharding.default_shards",
    "client_shards_seed": "data.tulu3_federated.sharding.seed",
    "client_shards_spec": "data.tulu3_federated.sharding.spec",
}

DATA_KEYS = {
    "data_root": "data.tulu3_federated.root",
}

DATA_VARIANT_ROOTS = {
    "group": "tulu3_federated_group",
    "source": "tulu3_federated_source",
}

BANK_KEYS = {
    "energy_target": "aggregator.unlearn.bank.energy_target",
    "r_global": "aggregator.unlearn.bank.r_global",
    "r_client": "aggregator.unlearn.bank.r_client",
    "beta_resid": "aggregator.unlearn.bank.beta_resid",
    "proj_rho": "train.unlearn.proj_rho",
}


def _load_wandb_config():
    raw = os.environ.get("WANDB_CONFIG")
    if raw:
        return json.loads(raw), os.environ.get("WANDB_RUN_ID"), None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "WANDB_CONFIG is not set and wandb is unavailable."
        ) from exc
    run = wandb.init()
    return dict(run.config), run.id, run


def _format_opts(config, mapping):
    opts = []
    for key, cfg_key in mapping.items():
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if isinstance(value, bool):
            value = str(value)
        opts.append(f"{cfg_key} {value}")
    return opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-script",
        default="scripts/run_tulu_pipeline.sh",
        help="Path to the pipeline script to submit.",
    )
    parser.add_argument(
        "--gpu-type",
        default="H200",
        help="GPU type to request for train/eval jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    cfg, sweep_run_id, run = _load_wandb_config()

    data_variant = cfg.pop("data_variant", None)
    if data_variant:
        root = DATA_VARIANT_ROOTS.get(str(data_variant))
        if not root:
            raise ValueError(
                f"Unsupported data_variant '{data_variant}', choose from "
                f"{sorted(DATA_VARIANT_ROOTS.keys())}")
        cfg["data_root"] = root

    common_opts = _format_opts(cfg, COMMON_KEYS)
    data_opts = _format_opts(cfg, DATA_KEYS)
    shard_opts = _format_opts(cfg, BASIC_SHARD_KEYS)
    bank_opts = _format_opts(cfg, BANK_KEYS)

    env = os.environ.copy()
    common_combined = common_opts + data_opts + shard_opts
    if common_combined:
        env["TULU_TRAIN_OPTS_COMMON"] = "::".join(common_combined)
    if bank_opts:
        env["TULU_TRAIN_OPTS_BANK"] = "::".join(bank_opts)

    if sweep_run_id:
        env["TULU_PIPE_ID"] = sweep_run_id
        env["WANDB_RUN_ID_BANK"] = sweep_run_id
    if run is not None:
        if run.project:
            env["WANDB_PROJECT"] = run.project
        if run.entity:
            env["WANDB_ENTITY"] = run.entity

    gpu_type = os.environ.get("TULU_GPU_TYPE", args.gpu_type)
    cmd = ["bash", args.pipeline_script, "--gpu-type", gpu_type]
    if args.dry_run:
        cmd.append("--dry-run")

    print("[sweep] submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
