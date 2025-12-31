#!/usr/bin/env python3
import argparse

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.configs.config import global_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for on-the-fly Tulu3 sharding.")
    parser.add_argument("--cfg", required=True, help="YAML config to load.")
    parser.add_argument(
        "--max-clients",
        type=int,
        default=20,
        help="Max client rows to print.")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional config overrides (key value pairs).")
    args = parser.parse_args()

    cfg = global_cfg.clone()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    setup_seed(cfg.seed)
    data, modified_cfg = get_data(config=cfg.clone(), client_cfgs=None)
    cfg.merge_from_other_cfg(modified_cfg)

    sharding_cfg = cfg.data.tulu3_federated.sharding
    print("sharding.enable:", sharding_cfg.enable)
    print("sharding.spec:", sharding_cfg.spec)
    print("sharding.default_shards:", sharding_cfg.default_shards)
    print("sharding.seed:", sharding_cfg.seed)

    client_ids = sorted([cid for cid in data.keys() if cid != 0])
    print("total_clients:", len(client_ids))
    for idx, cid in enumerate(client_ids[:args.max_clients], start=1):
        client = data[cid]
        train_len = len(client.train_data) if client.train_data else 0
        val_len = len(client.val_data) if client.val_data else 0
        print(f"{idx:02d}. client_id={cid} train={train_len} val={val_len}")
    if len(client_ids) > args.max_clients:
        print(f"... {len(client_ids) - args.max_clients} more clients")


if __name__ == "__main__":
    main()
