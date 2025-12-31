#!/usr/bin/env python3
"""Check Tulu3 federated data quality and assistant token coverage."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from federatedscope.llm.dataloader import get_tokenizer

# Allow running as a script without requiring `scripts` to be a package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from prepare_tulu3_federated import SampleProcessor  # noqa: E402


def _read_jsonl(path: Path, limit: int) -> List[Dict]:
    if not path.exists():
        return []
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            if limit > 0 and len(data) >= limit:
                break
    return data


def _init_processor(tokenizer, max_len: int) -> SampleProcessor:
    return SampleProcessor(
        tokenizer=tokenizer,
        max_length=max_len,
        min_assistant_tokens=1,
        trim_user_overflow=False,
        trim_user_token_limit=max_len,
        trim_overflow=False,
        trim_token_limit=max_len,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/tulu3_federated")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--tok-len", type=int, default=2048)
    parser.add_argument("--out", default="results_debug/tulu3_data_quality.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    manifest_path = root / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    tokenizer, _ = get_tokenizer(
        args.tokenizer, cache_dir=str(root.parent), tok_len=args.tok_len, pkg="huggingface_llm"
    )
    processor = _init_processor(tokenizer, args.tok_len)

    report = {
        "data_root": str(root),
        "max_samples_per_split": args.max_samples,
        "tok_len": args.tok_len,
        "clients": [],
    }

    for client in manifest.get("clients", []):
        name = client["name"]
        train_path = root / client["train_file"]
        val_path = root / client["val_file"]
        train_samples = _read_jsonl(train_path, args.max_samples)
        val_samples = _read_jsonl(val_path, args.max_samples)
        stats = {
            "client": name,
            "train_checked": len(train_samples),
            "val_checked": len(val_samples),
            "invalid": 0,
            "reasons": {},
            "assistant_tokens_total": 0,
            "assistant_tokens_avg": 0.0,
        }

        total_valid = 0
        for split_samples in (train_samples, val_samples):
            for sample in split_samples:
                processed, meta = processor.process(sample)
                reason = meta.get("reason")
                if processed is None:
                    stats["invalid"] += 1
                    stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                    continue
                total_valid += 1
                stats["assistant_tokens_total"] += int(meta.get("assistant_tokens", 0))

        if total_valid > 0:
            stats["assistant_tokens_avg"] = stats["assistant_tokens_total"] / total_valid

        report["clients"].append(stats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote data quality report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
