#!/usr/bin/env python3
"""Scan a checkpoint for NaN/Inf tensors to diagnose GSM8K eval crashes."""

import argparse
import sys
from typing import Any, Dict, Iterable, Tuple

import torch


def _iter_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(value, name)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            name = f"{prefix}[{idx}]"
            yield from _iter_tensors(value, name)


def scan_checkpoint(path: str) -> Dict[str, int]:
    ckpt = torch.load(path, map_location="cpu")
    bad = []
    total = 0
    for name, tensor in _iter_tensors(ckpt):
        total += 1
        if not torch.is_floating_point(tensor):
            continue
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            bad.append(name)
    return {"total_tensors": total, "bad_tensors": len(bad), "bad_names": bad}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    args = parser.parse_args()

    results = scan_checkpoint(args.ckpt)
    print(f"Scanned {results['total_tensors']} tensors.")
    print(f"Found {results['bad_tensors']} tensors with NaN/Inf.")
    if results["bad_names"]:
        print("Example bad tensor keys:")
        for name in results["bad_names"][:20]:
            print(f"  - {name}")
        if results["bad_tensors"] > 20:
            print(f"  ... {results['bad_tensors'] - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
