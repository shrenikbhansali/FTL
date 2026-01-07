#!/usr/bin/env python3
"""Prepare federated splits from P3 (PromptSource / T0 mixture).

This script downloads P3 from Hugging Face, converts each record into a
chat-style sample with `messages`, and writes per-client train/val JSONL
files plus a manifest compatible with the tulu3_federated loader.
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return safe.strip("_") or "client"


def _pick_first(text_or_list: Any) -> str:
    if isinstance(text_or_list, list) and text_or_list:
        return str(text_or_list[0])
    if text_or_list is None:
        return ""
    return str(text_or_list)


def _estimate_length(tokenizer, messages, max_len: int) -> int:
    if tokenizer is None:
        return 0
    if hasattr(tokenizer, "apply_chat_template"):
        tokens = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_len,
            add_generation_prompt=False,
        )
        return int(tokens.shape[1])
    joined = "\n\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in messages
    )
    tokens = tokenizer(joined,
                       return_tensors="pt",
                       padding=False,
                       truncation=True,
                       max_length=max_len)
    return int(tokens["input_ids"].shape[1])


def _build_messages(user_text: str, assistant_text: str) -> list:
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def _select_group_key(record: Dict[str, Any], group_by: str) -> str:
    if group_by == "config" and record.get("config_name"):
        return record["config_name"]
    if group_by == "dataset" and record.get("dataset_name"):
        return record["dataset_name"]
    if group_by == "category" and record.get("category"):
        return record["category"]
    if record.get("dataset_name"):
        return record["dataset_name"]
    if record.get("config_name"):
        return record["config_name"]
    return "unknown_task"


def _load_config_list(dataset: str,
                      configs_arg: Optional[str],
                      config_file: Optional[str]) -> List[str]:
    if configs_arg:
        return [item.strip() for item in configs_arg.split(",") if item.strip()]
    if config_file:
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
            raise ValueError("Config JSON must be a list of config names.")
        configs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                configs.append(line)
        if configs:
            return configs
    return get_dataset_config_names(dataset)


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


def _load_category_rules(path: Optional[str]) -> List[Tuple[re.Pattern, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Category map not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rules: List[Tuple[re.Pattern, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            rules.append((re.compile(rf"^{re.escape(str(key))}"), str(value)))
        return rules
    if isinstance(payload, list):
        for entry in payload:
            pattern = entry.get("pattern")
            category = entry.get("category")
            if not pattern or not category:
                continue
            rules.append((re.compile(pattern), str(category)))
        return rules
    raise ValueError("Category map must be a JSON dict or list of {pattern, category}.")


def _match_category(dataset_name: str,
                    rules: List[Tuple[re.Pattern, str]]) -> str:
    for pattern, category in rules:
        if pattern.search(dataset_name):
            return category
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="bigscience/P3",
                    help="HF dataset id for P3.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir",
                    default="data/p3_federated",
                    help="Directory to write federated clients + manifest.")
    ap.add_argument("--group-by",
                    choices=["config", "dataset", "category"],
                    default="dataset",
                    help="Client grouping key.")
    ap.add_argument("--configs",
                    default="",
                    help="Comma-separated list of P3 config names.")
    ap.add_argument("--config-file",
                    default=None,
                    help="Optional file with one config name per line (or JSON list).")
    ap.add_argument("--category-map",
                    default=None,
                    help="Optional JSON mapping for dataset prefixes -> category.")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-cache-dir",
                    default=os.environ.get("HF_HOME"),
                    help="Optional HF cache directory inside scratch.")
    ap.add_argument("--tokenizer",
                    default="meta-llama/Llama-2-7b-hf",
                    help="Tokenizer for length filtering.")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--drop-long", action="store_true", default=True)
    ap.add_argument("--no-drop-long", action="store_false", dest="drop_long")
    ap.add_argument("--max-total-samples", type=int, default=200000)
    ap.add_argument("--max-samples-per-client", type=int, default=20000)
    ap.add_argument("--max-clients", type=int, default=None)
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", action="store_false", dest="streaming")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--overwrite",
                    action="store_true",
                    help="Remove existing output directory before writing.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            model_max_length=args.max_length,
            use_fast=False,
            cache_dir=args.hf_cache_dir,
        )

    rng = random.Random(args.seed)

    configs = _load_config_list(args.dataset, args.configs, args.config_file)
    if not configs:
        raise ValueError("No P3 configs were provided or discovered.")
    dataset_name_map = _infer_dataset_prefixes(configs)
    category_rules = _load_category_rules(args.category_map) if args.group_by == "category" else []

    grouped: Dict[str, list] = defaultdict(list)
    manifest_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    manifest_sources: Dict[str, str] = {}
    total_seen = 0
    total_written = 0

    for config_name in configs:
        if args.max_total_samples is not None and total_written >= args.max_total_samples:
            break
        try:
            dataset = load_dataset(args.dataset,
                                   config_name,
                                   split=args.split,
                                   streaming=args.streaming,
                                   cache_dir=args.hf_cache_dir)
        except Exception as exc:
            print(f"[prepare_p3] skipping config {config_name}: {exc}")
            continue
        dataset_name = dataset_name_map.get(config_name, config_name)
        if args.group_by == "category":
            category = _match_category(dataset_name, category_rules)
        else:
            category = None

        for record in dataset:
            total_seen += 1
            if args.max_records is not None and total_seen > args.max_records:
                break
            if args.max_total_samples is not None and total_written >= args.max_total_samples:
                break
            user_text = _pick_first(
                record.get("inputs_pretokenized", record.get("inputs", ""))
            )
            assistant_text = _pick_first(
                record.get("targets_pretokenized", record.get("targets", ""))
            )
            if not (user_text or assistant_text):
                continue

            messages = _build_messages(str(user_text), str(assistant_text))
            if tokenizer is not None and args.drop_long:
                length = _estimate_length(tokenizer, messages, args.max_length)
                if length > args.max_length:
                    continue

            payload = {
                "messages": messages,
                "config_name": config_name,
                "dataset_name": dataset_name,
                "category": category,
                "source": config_name,
            }
            group_value = _select_group_key(payload, args.group_by)
            if args.group_by == "category" and group_value == "unknown":
                continue
            client_name = _sanitize_name(group_value)
            if args.max_clients is not None and client_name not in manifest_counts:
                if len(manifest_counts) >= args.max_clients:
                    continue

            if args.streaming:
                is_val = rng.random() < args.val_fraction
                split_name = "val" if is_val else "train"
                if args.max_samples_per_client is not None and \
                        manifest_counts[client_name][split_name] >= args.max_samples_per_client:
                    continue
                manifest_counts[client_name][split_name] += 1
                manifest_sources.setdefault(client_name, group_value)

                split_path = output_dir / client_name / f"{split_name}.jsonl"
                split_path.parent.mkdir(parents=True, exist_ok=True)
                with split_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False))
                    f.write("\n")
                total_written += 1
            else:
                grouped[client_name].append(payload)

    if not args.streaming:
        for client_name, records in grouped.items():
            rng.shuffle(records)
            if args.max_total_samples is not None:
                remaining = max(args.max_total_samples - total_written, 0)
                records = records[:remaining]
            if args.max_samples_per_client is not None:
                records = records[:args.max_samples_per_client]
            if not records:
                continue
            split_idx = max(1, int((1 - args.val_fraction) * len(records)))
            train_records = records[:split_idx]
            val_records = records[split_idx:]
            _write_jsonl(output_dir / client_name / "train.jsonl", train_records)
            _write_jsonl(output_dir / client_name / "val.jsonl", val_records)
            manifest_counts[client_name]["train"] = len(train_records)
            manifest_counts[client_name]["val"] = len(val_records)
            manifest_sources[client_name] = client_name
            total_written += len(records)
            if args.max_total_samples is not None and total_written >= args.max_total_samples:
                break

    manifest_clients = []
    for client_name, counts in manifest_counts.items():
        manifest_clients.append({
            "name": client_name,
            "group_key": args.group_by,
            "group_values": [manifest_sources.get(client_name, client_name)],
            "train_examples": counts["train"],
            "val_examples": counts["val"],
            "train_file": f"{client_name}/train.jsonl",
            "val_file": f"{client_name}/val.jsonl",
            "source": manifest_sources.get(client_name, client_name),
        })

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "val_fraction": args.val_fraction,
        "group_by": args.group_by,
        "clients": manifest_clients,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(manifest_clients)} clients to {output_dir}")


if __name__ == "__main__":
    main()
