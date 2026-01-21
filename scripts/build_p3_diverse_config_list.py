#!/usr/bin/env python3
"""Build a diverse P3 config list (one config per dataset prefix).

Uses the same prefix inference logic as scripts/prepare_p3_federated.py.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import get_dataset_config_names


def _infer_dataset_prefixes(configs: List[str]) -> Dict[str, str]:
    prefix_counts: Dict[str, int] = defaultdict(int)
    for config in configs:
        parts = config.split("_")
        for i in range(1, len(parts)):
            prefix = "_".join(parts[:i])
            prefix_counts[prefix] += 1
    dataset_map = {}
    for config in configs:
        parts = config.split("_")
        chosen = config
        for i in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:i])
            if prefix_counts.get(prefix, 0) > 1:
                chosen = prefix
                break
        dataset_map[config] = chosen
    return dataset_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigscience/P3")
    ap.add_argument("--output",
                    default="materials/p3_config_diverse.txt",
                    help="Output file to write config names.")
    ap.add_argument("--max-per-prefix",
                    type=int,
                    default=1,
                    help="How many configs to keep per dataset prefix.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle",
                    action="store_true",
                    help="Shuffle configs before selecting per prefix.")
    args = ap.parse_args()

    configs = get_dataset_config_names(args.dataset)
    if not configs:
        raise ValueError(f"No configs found for dataset {args.dataset}")
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(configs)
    else:
        configs = sorted(configs)

    dataset_map = _infer_dataset_prefixes(configs)
    per_prefix: Dict[str, List[str]] = defaultdict(list)
    for cfg in configs:
        per_prefix[dataset_map[cfg]].append(cfg)

    chosen = []
    max_per = max(1, int(args.max_per_prefix))
    for prefix in sorted(per_prefix):
        pool = per_prefix[prefix]
        if not args.shuffle:
            pool = sorted(pool)
        chosen.extend(pool[:max_per])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for cfg in chosen:
            f.write(cfg + "\n")

    print(f"[p3-diverse] dataset={args.dataset}")
    print(f"[p3-diverse] prefixes={len(per_prefix)}")
    print(f"[p3-diverse] configs={len(chosen)}")
    print(f"[p3-diverse] wrote {out_path}")


if __name__ == "__main__":
    main()
