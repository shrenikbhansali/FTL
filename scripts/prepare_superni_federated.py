#!/usr/bin/env python3
"""Prepare federated splits from Super-NaturalInstructions (SuperNI).

This script downloads a SuperNI-style dataset from Hugging Face, converts
each instance into a chat-style sample with `messages`, and writes per-client
train/val JSONL files plus a manifest compatible with the tulu3_federated
loader. It supports both task-level datasets (with "instances") and
row-level datasets (with "task_name", "inputs", "targets").
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("_") or "client"


def _pick_first(text_or_list: Any) -> str:
    if isinstance(text_or_list, list) and text_or_list:
        return str(text_or_list[0])
    if text_or_list is None:
        return ""
    return str(text_or_list)


def _build_messages(definition: Optional[str],
                    user_text: str,
                    assistant_text: str) -> List[Dict[str, str]]:
    messages = []
    if definition:
        messages.append({"role": "system", "content": definition})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _estimate_length(tokenizer, messages: List[Dict[str, str]], max_len: int) -> int:
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
    # Fallback: join with role prefixes.
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


def _iter_superni_tasks(dataset: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for task in dataset:
        instances = task.get("instances", [])
        if not isinstance(instances, list):
            continue
        for inst in instances:
            yield task, inst


def _iter_superni_rows(dataset: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in dataset:
        yield row


def _peek_first(dataset: Iterable[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Iterator[Dict[str, Any]]]:
    iterator = iter(dataset)
    first = next(iterator, None)
    if first is None:
        return None, iter(())
    def _chain():
        yield first
        for item in iterator:
            yield item
    return first, _chain()


def _format_definition(task: Dict[str, Any], include_pos_examples: bool) -> str:
    definition = task.get("definition", [])
    if isinstance(definition, list):
        definition_text = "\n".join(str(item).strip() for item in definition if item)
    else:
        definition_text = str(definition).strip() if definition else ""

    if include_pos_examples:
        examples = task.get("positive_examples", [])
        if isinstance(examples, list):
            example_lines = []
            for ex in examples:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                if inp or out:
                    example_lines.append(f"Input: {inp}\nOutput: {out}")
            if example_lines:
                definition_text = "\n\n".join(
                    [definition_text, "Examples:", "\n\n".join(example_lines)]
                    if definition_text else ["Examples:", "\n\n".join(example_lines)]
                )
    return definition_text.strip()


def _write_jsonl(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="Muennighoff/natural-instructions",
                    help="HF dataset id for SuperNI-style data.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir",
                    default="data/superni_federated",
                    help="Directory to write federated clients + manifest.")
    ap.add_argument("--group-by",
                    choices=["task_name", "task_category"],
                    default="task_name",
                    help="Client grouping key.")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-cache-dir",
                    default=os.environ.get("HF_HOME"),
                    help="Optional HF cache directory inside scratch.")
    ap.add_argument("--dedupe-by",
                    choices=["none", "id", "inputs"],
                    default="id",
                    help="Optional deduplication key for row-style datasets.")
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
    ap.add_argument("--include-positive-examples", action="store_true")
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

    dataset = load_dataset(args.dataset,
                           split=args.split,
                           streaming=args.streaming,
                           cache_dir=args.hf_cache_dir)
    first, dataset_iter = _peek_first(dataset)
    if first is None:
        print(f"No records found in {args.dataset}:{args.split}")
        return
    is_task_mode = "instances" in first
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    manifest_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0})
    client_sources: Dict[str, str] = {}
    total_written = 0
    seen_hashes: Optional[set] = set() if args.dedupe_by != "none" else None

    if is_task_mode:
        task_iter = _iter_superni_tasks(dataset_iter)
        for task, inst in task_iter:
            task_name = task.get("task_name", "unknown_task")
            task_category = task.get("task_category", "unknown_category")
            group_value = task_name if args.group_by == "task_name" else task_category
            client_name = _sanitize_name(group_value)
            if args.max_clients is not None and client_name not in manifest_counts:
                if len(manifest_counts) >= args.max_clients:
                    continue
            definition = _format_definition(task, args.include_positive_examples)
            user_text = _pick_first(inst.get("input", ""))
            assistant_text = _pick_first(inst.get("output", ""))
            if not (user_text or assistant_text):
                continue

            messages = _build_messages(definition, user_text, assistant_text)
            if tokenizer is not None and args.drop_long:
                length = _estimate_length(tokenizer, messages, args.max_length)
                if length > args.max_length:
                    continue

            record = {
                "messages": messages,
                "task_name": task_name,
                "task_category": task_category,
                "source": task_name,
            }

            if args.streaming:
                if args.max_total_samples is not None and total_written >= args.max_total_samples:
                    break
                is_val = rng.random() < args.val_fraction
                split_name = "val" if is_val else "train"
                if args.max_samples_per_client is not None and \
                        manifest_counts[client_name][split_name] >= args.max_samples_per_client:
                    continue
                split_path = output_dir / client_name / f"{split_name}.jsonl"
                split_path.parent.mkdir(parents=True, exist_ok=True)
                with split_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                manifest_counts[client_name][split_name] += 1
                client_sources.setdefault(client_name, group_value)
                total_written += 1
            else:
                grouped[group_value].append(record)

    else:
        row_iter = _iter_superni_rows(dataset_iter)
        for row in row_iter:
            task_name = row.get("task_name", "unknown_task")
            task_category = row.get("task_category") or task_name
            group_value = task_name if args.group_by == "task_name" else task_category
            client_name = _sanitize_name(group_value)
            if args.max_clients is not None and client_name not in manifest_counts:
                if len(manifest_counts) >= args.max_clients:
                    continue
            definition = row.get("definition") or ""
            user_text = _pick_first(row.get("inputs", row.get("input", "")))
            assistant_text = _pick_first(row.get("targets", row.get("output", "")))
            if not (user_text or assistant_text):
                continue

            if seen_hashes is not None:
                if args.dedupe_by == "id":
                    raw_key = row.get("id")
                elif args.dedupe_by == "inputs":
                    raw_key = user_text
                else:
                    raw_key = None
                if raw_key:
                    digest = hashlib.sha1(str(raw_key).encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)

            messages = _build_messages(definition, user_text, assistant_text)
            if tokenizer is not None and args.drop_long:
                length = _estimate_length(tokenizer, messages, args.max_length)
                if length > args.max_length:
                    continue

            record = {
                "messages": messages,
                "task_name": task_name,
                "task_category": task_category,
                "source": task_name,
            }

            if args.streaming:
                if args.max_total_samples is not None and total_written >= args.max_total_samples:
                    break
                is_val = rng.random() < args.val_fraction
                split_name = "val" if is_val else "train"
                if args.max_samples_per_client is not None and \
                        manifest_counts[client_name][split_name] >= args.max_samples_per_client:
                    continue
                split_path = output_dir / client_name / f"{split_name}.jsonl"
                split_path.parent.mkdir(parents=True, exist_ok=True)
                with split_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                manifest_counts[client_name][split_name] += 1
                client_sources.setdefault(client_name, group_value)
                total_written += 1
            else:
                grouped[group_value].append(record)

    if not args.streaming:
        for raw_key, records in grouped.items():
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

            client_name = _sanitize_name(raw_key)
            train_path = Path(client_name) / "train.jsonl"
            val_path = Path(client_name) / "val.jsonl"
            _write_jsonl(output_dir / train_path, train_records)
            _write_jsonl(output_dir / val_path, val_records)
            manifest_counts[client_name]["train"] = len(train_records)
            manifest_counts[client_name]["val"] = len(val_records)
            client_sources[client_name] = raw_key
            total_written += len(records)
            if args.max_total_samples is not None and total_written >= args.max_total_samples:
                break

    manifest_clients = []
    for client_name, counts in manifest_counts.items():
        manifest_clients.append({
            "name": client_name,
            "group_key": args.group_by,
            "group_values": [client_sources.get(client_name, client_name)],
            "train_examples": counts["train"],
            "val_examples": counts["val"],
            "train_file": f"{client_name}/train.jsonl",
            "val_file": f"{client_name}/val.jsonl",
            "source": client_sources.get(client_name, client_name),
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
