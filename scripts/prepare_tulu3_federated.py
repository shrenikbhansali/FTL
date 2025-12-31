#!/usr/bin/env python3
"""Prepare federated splits from the Tulu 3 SFT mixture.

This script downloads ``allenai/tulu-3-sft-mixture`` with Hugging Face
``datasets`` and partitions it into task-family-specific shards. Each
shard represents a federated client for FS-LLM experiments. The mapping
between mixture sources and high-level task families is derived from the
Open-Instruct configs and data scripts (see ``scripts/data`` within the
Open-Instruct repo for references).

Usage example::

    python scripts/prepare_tulu3_federated.py \
        --output-dir data/tulu3_federated \
        --client-config materials/tulu3_clients.yaml

To group by constituent dataset sources instead of task families::

    python scripts/prepare_tulu3_federated.py \
        --output-dir data/tulu3_federated_by_source \
        --group-by source

You can shard each task family into multiple clients by adding
``num_shards`` (or ``shard_names``) per entry in the client config, e.g.::

    clients:
      - name_prefix: chat_client
        families: [chat]
        num_shards: 3
      - name_prefix: qa_client
        families: [qa]
        num_shards: 3

The produced directory contains one sub-directory per client with
``train.jsonl``/``val.jsonl`` files and a manifest describing the
configuration.
"""

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from datasets import load_dataset

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None

# ``SOURCE_TO_FAMILY`` is curated from the Open-Instruct configs detailing
# the Tulu 3 mixture components. Each entry references a documented dataset:
# - Persona-driven math/code/IF data: ``scripts/persona_driven_data_gen``
# - Safety datasets: ``scripts/data/get_statistics_tulu_v3.sh``
# - FLAN / Aya / TableGPT / SciRIFF: ``configs/train_configs/sft``
# These sources correspond to the 19 mixture identifiers listed on the
# HuggingFace card ``allenai/tulu-3-sft-mixture``.
SOURCE_TO_FAMILY: Dict[str, str] = {
    # Open-domain instruction/chat style data.
    "ai2-adapt-dev/oasst1_converted": "chat",  # OpenAssistant 1 format.
    "ai2-adapt-dev/tulu_hard_coded_repeated_10": "chat",  # curated prompts.
    "ai2-adapt-dev/no_robots_converted": "chat",  # No Robots dialogue set.
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": "chat",  # WildChat conversations.

    # Knowledge and QA heavy corpora (FLAN instructions + TableGPT).
    "ai2-adapt-dev/flan_v2_converted": "qa",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k": "qa",

    # Multilingual / translation focused Aya dataset.
    "ai2-adapt-dev/tulu_v3.9_aya_100k": "multi",

    # General reasoning / precise instruction following datasets.
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980": "reasoning",
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k": "reasoning",

    # Math-focused persona and benchmark blends.
    "ai2-adapt-dev/personahub_math_v5_regen_149960": "math",
    "allenai/tulu-3-sft-personas-math-grade": "math",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": "math",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": "math",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "math",

    # Code generation datasets.
    "ai2-adapt-dev/personahub_code_v2_34999": "code",
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": "code",

    # Safety / refusal / jailbreak corpora.
    "ai2-adapt-dev/coconot_converted": "safety",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k": "safety",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k":
    "safety",
}

FAMILY_CHOICES = {
    "chat", "qa", "reasoning", "math", "code", "safety", "multi"
}

DEFAULT_CLIENTS = [
    {"name": "chat_client", "families": ["chat"]},
    {"name": "qa_client", "families": ["qa"]},
    {"name": "reasoning_client", "families": ["reasoning"]},
    {"name": "math_client", "families": ["math"]},
    {"name": "code_client", "families": ["code"]},
    {"name": "safety_client", "families": ["safety"]},
    {"name": "multilingual_client", "families": ["multi"]},
]


class SampleProcessor:
    """Detect and optionally fix samples that lose assistant labels."""

    _ROLE_ALIASES = {
        "human": "user",
        "ai": "assistant",
    }

    def __init__(self,
                 tokenizer,
                 max_length: int,
                 min_assistant_tokens: int,
                 trim_user_overflow: bool,
                 trim_user_token_limit: int,
                 trim_overflow: bool,
                 trim_token_limit: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = max(min_assistant_tokens, 1)
        self.trim_user_overflow = trim_user_overflow
        self.trim_user_token_limit = trim_user_token_limit
        self.trim_overflow = trim_overflow
        self.trim_token_limit = trim_token_limit

    def process(self, record: Dict) -> Tuple[Optional[Dict], Dict]:
        messages = record.get("messages", [])
        messages, normalize_reason = self._normalize_messages(messages)
        meta = {
            "assistant_tokens": 0,
            "trimmed": False,
            "reason": normalize_reason,
        }
        if messages is None:
            return None, meta

        label_tokens = self._count_assistant_tokens(messages)
        meta["assistant_tokens"] = label_tokens
        if label_tokens >= self.min_assistant_tokens:
            record = copy.deepcopy(record)
            record["messages"] = messages
            return record, meta

        if self.trim_user_overflow:
            user_trimmed_messages, did_trim = self._trim_user_turns(messages)
            if did_trim:
                new_tokens = self._count_assistant_tokens(user_trimmed_messages)
                if new_tokens >= self.min_assistant_tokens:
                    new_record = copy.deepcopy(record)
                    new_record["messages"] = user_trimmed_messages
                    meta["trimmed"] = True
                    meta["assistant_tokens"] = new_tokens
                    meta["reason"] = None
                    return new_record, meta

        if self.trim_overflow:
            trimmed_messages, did_trim = self._trim_final_assistant(messages)
            if did_trim:
                new_tokens = self._count_assistant_tokens(trimmed_messages)
                if new_tokens >= self.min_assistant_tokens:
                    new_record = copy.deepcopy(record)
                    new_record["messages"] = trimmed_messages
                    meta["trimmed"] = True
                    meta["assistant_tokens"] = new_tokens
                    meta["reason"] = None
                    return new_record, meta

        meta["reason"] = "insufficient_assistant_tokens"
        return None, meta

    def _trim_user_turns(self, messages: List[Dict]) -> Tuple[List[Dict], bool]:
        trimmed = False
        new_messages = copy.deepcopy(messages)
        for msg in new_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            tokens = self.tokenizer.encode(content,
                                           add_special_tokens=False)
            if len(tokens) <= self.trim_user_token_limit:
                continue
            tokens = tokens[-self.trim_user_token_limit:]
            msg["content"] = self.tokenizer.decode(tokens,
                                                   skip_special_tokens=True)
            trimmed = True
        return new_messages, trimmed

    def _normalize_messages(self,
                            messages: List[Dict]) -> Tuple[Optional[List[Dict]], Optional[str]]:
        if not isinstance(messages, list) or len(messages) == 0:
            return None, "missing_messages"

        normalized: List[Dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            role = self._ROLE_ALIASES.get(role, role)
            content = msg.get("content", "")
            if not role:
                continue
            if role == "assistant" and not normalized:
                role = "system"

            if role == "system":
                if normalized and normalized[0].get("role") == "system":
                    normalized[0]["content"] = self._merge_content(
                        normalized[0].get("content", ""), content)
                else:
                    normalized.insert(0, {"role": "system", "content": content})
                continue

            if normalized and normalized[-1].get("role") == role:
                normalized[-1]["content"] = self._merge_content(
                    normalized[-1].get("content", ""), content)
            else:
                normalized.append({"role": role, "content": content})

        if not normalized:
            return None, "empty_after_normalize"

        start_idx = 1 if normalized[0].get("role") == "system" else 0
        roles = [m.get("role") for m in normalized[start_idx:]]
        if "assistant" not in roles:
            return None, "no_assistant_turn"
        if not roles or roles[0] != "user":
            return None, "invalid_role_sequence"
        for prev, curr in zip(roles, roles[1:]):
            if prev == curr:
                return None, "invalid_role_sequence"
            if curr not in ("user", "assistant"):
                return None, "invalid_role_sequence"
        return normalized, None

    def _merge_content(self, left: str, right: str) -> str:
        if left and right:
            return f"{left}\n\n{right}"
        return left or right

    def _trim_final_assistant(self, messages: List[Dict]) -> Tuple[List[Dict], bool]:
        trimmed = False
        new_messages = copy.deepcopy(messages)
        for idx in range(len(new_messages) - 1, -1, -1):
            msg = new_messages[idx]
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            tokens = self.tokenizer.encode(content,
                                           add_special_tokens=False)
            if len(tokens) <= self.trim_token_limit:
                continue
            tokens = tokens[-self.trim_token_limit:]
            msg["content"] = self.tokenizer.decode(tokens,
                                                    skip_special_tokens=True)
            trimmed = True
            break
        return new_messages, trimmed

    def _count_assistant_tokens(self, messages: List[Dict]) -> int:
        encoding = self._tokenize_messages(messages)
        labels = encoding["labels"]
        return int((labels != -100).sum().item())

    def _tokenize_messages(self, messages: List[Dict]):
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_generation_prompt=False,
            )
        else:
            raise RuntimeError(
                "The selected tokenizer does not support chat templates.")

        labels = input_ids.clone()
        for idx, message in enumerate(messages):
            if message.get("role") == "assistant":
                continue
            start = self._measure_prefix(messages[:idx])
            end = self._measure_prefix(
                messages[:idx + 1],
                add_generation_prompt=(idx < len(messages) - 1 and
                                       messages[idx + 1].get("role") ==
                                       "assistant"))
            labels[:, start:end] = -100
            if self.max_length and end >= self.max_length:
                break
        return {
            "input_ids": input_ids[:, :self.max_length],
            "labels": labels[:, :self.max_length],
        }

    def _measure_prefix(self,
                        conversation_slice: List[Dict],
                        add_generation_prompt: bool = False) -> int:
        if len(conversation_slice) == 0:
            return 0
        tokens = self.tokenizer.apply_chat_template(
            conversation=conversation_slice,
            tokenize=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=add_generation_prompt,
        )
        return tokens.shape[1]


class StatsRecorder:
    def __init__(self, family_counts: Dict[str, int], output_path: Path):
        self.family_counts = family_counts
        self.output_path = output_path
        self.client_stats: Dict[str, Dict] = {}
        self.family_post_counts = Counter()
        self.drop_reasons = Counter()

    def register_client(self, name: str, families: List[str], total: int):
        self.client_stats[name] = {
            "families": sorted(families),
            "total_before": total,
            "splits": {}
        }

    def record_split(self, name: str, split: str, before: int, kept: int,
                     dropped: int, trimmed: int, reasons: Counter):
        entry = self.client_stats[name]["splits"].setdefault(split, {})
        entry.update({
            "before": before,
            "kept": kept,
            "dropped": dropped,
            "trimmed": trimmed,
        })
        self.drop_reasons.update(reasons)

    def accumulate_family_counts(self, families: List[str], kept_total: int):
        for fam in families:
            self.family_post_counts[fam] += kept_total

    def write(self):
        lines = []
        lines.append("=== Preprocessing statistics ===")
        lines.append("-- Family counts before filtering --")
        for fam, cnt in sorted(self.family_counts.items()):
            lines.append(f"{fam}: {cnt}")
        lines.append("")

        for name in sorted(self.client_stats):
            info = self.client_stats[name]
            lines.append(f"Client {name} (families: {', '.join(info['families'])})")
            lines.append(f"  Total before filtering: {info['total_before']}")
            for split in ["train", "val"]:
                if split not in info["splits"]:
                    continue
                split_info = info["splits"][split]
                lines.append(
                    f"  {split.capitalize()}: before={split_info['before']} kept={split_info['kept']} dropped={split_info['dropped']} trimmed={split_info['trimmed']}")
            lines.append("")

        lines.append("-- Family counts after filtering (train+val) --")
        for fam, cnt in sorted(self.family_post_counts.items()):
            lines.append(f"{fam}: {cnt}")

        if self.drop_reasons:
            lines.append("")
            lines.append("-- Drop reasons --")
            for reason, cnt in self.drop_reasons.items():
                lines.append(f"{reason}: {cnt}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines), encoding="utf-8")


def build_sample_processor(args) -> Optional[SampleProcessor]:
    if not args.enable_length_filter:
        return None
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for --enable-length-filter; install torch or "
            "disable the filter.")
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for --enable-length-filter; install it "
            "or disable the filter.")
    cache_dir = args.filter_tokenizer_cache or args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        args.filter_tokenizer_name,
        cache_dir=cache_dir,
        model_max_length=args.filter_max_length,
        padding_side="right",
        use_fast=False,
    )
    name_lower = args.filter_tokenizer_name.lower()
    if "llama" in name_lower:
        tokenizer.truncation_side = "left"
    return SampleProcessor(tokenizer=tokenizer,
                           max_length=args.filter_max_length,
                           min_assistant_tokens=args.filter_min_assistant_tokens,
                           trim_user_overflow=args.filter_trim_user_overflow,
                           trim_user_token_limit=args.filter_user_tail_tokens,
                           trim_overflow=args.filter_trim_overflow,
                           trim_token_limit=args.filter_assistant_tail_tokens)


def load_client_config(path: Path) -> Dict[str, Sequence[Dict[str, Sequence[str]]]]:
    if path is None:
        return {"clients": DEFAULT_CLIENTS}

    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as err:  # pragma: no cover - informative error
            raise RuntimeError(
                "PyYAML is required to parse YAML configs. Install via"
                " `pip install pyyaml`." ) from err
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict) or "clients" not in data:
        raise ValueError("Client config must define a top-level 'clients' list")
    return data


def _parse_families(spec: Dict) -> List[str]:
    families = spec.get("families")
    if families is None:
        families = spec.get("family")
    if families is None:
        raise ValueError(f"Client spec is missing families: {spec}")
    if isinstance(families, str):
        families = [families]
    if not isinstance(families, (list, tuple)) or not families:
        raise ValueError(f"Client spec families must be a non-empty list: {spec}")
    return list(families)


def _normalize_shard_names(spec: Dict, num_shards: int) -> List[str]:
    shard_names = spec.get("shard_names")
    if shard_names is None:
        shard_names = spec.get("names")
    if shard_names is None:
        return []
    if not isinstance(shard_names, (list, tuple)):
        raise ValueError(
            f"shard_names must be a list when provided: {spec}")
    if num_shards is not None and len(shard_names) != num_shards:
        raise ValueError(
            f"shard_names length ({len(shard_names)}) does not match num_shards "
            f"({num_shards}) in {spec}")
    return list(shard_names)


def normalize_client_specs(raw_clients: Sequence[Dict], filter_key: str) -> List[Dict]:
    groups = []
    for spec in raw_clients:
        if not isinstance(spec, dict):
            raise ValueError(f"Client spec must be a dict: {spec}")
        if filter_key == "task_family":
            labels = _parse_families(spec)
        else:
            labels = spec.get("sources")
            if labels is None:
                raise ValueError(
                    f"Client spec is missing sources for group-by {filter_key}: {spec}")
            if isinstance(labels, str):
                labels = [labels]
            if not isinstance(labels, (list, tuple)) or not labels:
                raise ValueError(
                    f"Client spec sources must be a non-empty list: {spec}")
        num_shards = spec.get("num_shards")
        if num_shards is None:
            num_shards = spec.get("shards")
        shard_names = _normalize_shard_names(spec, num_shards)
        if num_shards is None:
            num_shards = len(shard_names) if shard_names else 1
        if not isinstance(num_shards, int) or num_shards < 1:
            raise ValueError(f"num_shards must be a positive int: {spec}")

        if num_shards == 1:
            name = spec.get("name") or (shard_names[0] if shard_names else None)
            if not name:
                raise ValueError(f"Single client spec missing name: {spec}")
            names = [name]
        else:
            if shard_names:
                names = shard_names
            else:
                prefix = spec.get("name_prefix") or spec.get("name")
                if not prefix:
                    raise ValueError(
                        f"Sharded client spec missing name_prefix/name: {spec}")
                names = [f"{prefix}_{idx + 1}" for idx in range(num_shards)]

        groups.append({
            "labels": list(labels),
            "names": names,
            "filter_key": filter_key,
        })
    return groups


def _sanitize_source_name(source: str, max_len: int = 64) -> str:
    base = source.split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    if not slug:
        slug = "source"
    if len(slug) > max_len:
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:max_len - 9]}_{digest}"
    return slug


def build_source_groups(sources: Sequence[str]) -> List[Dict]:
    used = set()
    groups = []
    for src in sources:
        name = _sanitize_source_name(src)
        if name in used:
            digest = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
            name = f"{name}_{digest}"
        used.add(name)
        groups.append({
            "labels": [src],
            "names": [name],
            "filter_key": "source",
        })
    return groups


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already exists. Use --overwrite to clobber.")
    path.mkdir(parents=True, exist_ok=True)


def annotate_task_family(example):
    source = example["source"]
    try:
        example["task_family"] = SOURCE_TO_FAMILY[source]
    except KeyError as exc:  # pragma: no cover - protects against new sources
        raise KeyError(
            f"Missing source->family mapping for '{source}'. Update"
            " SOURCE_TO_FAMILY before proceeding." ) from exc
    return example


def downsample(dataset, limit: int, seed: int):
    if limit is None or len(dataset) <= limit:
        return dataset
    dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(limit))


def train_val_split(dataset, val_frac: float, seed: int):
    """Return train/val ``Dataset`` pairs using deterministic shuffling."""
    if len(dataset) == 0:
        return dataset, dataset.select([])
    dataset = dataset.shuffle(seed=seed)
    val_count = int(round(len(dataset) * val_frac))
    if val_count >= len(dataset) and len(dataset) > 1:
        val_count = len(dataset) - 1
    if len(dataset) == 1:
        val_count = 0
    val_ids = range(val_count)
    train_ids = range(val_count, len(dataset))
    return dataset.select(train_ids), dataset.select(val_ids)


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def process_and_save_split(split_ds,
                           path: Path,
                           processor: Optional[SampleProcessor]):
    """
    Apply optional preprocessing to a dataset split and write JSONL output.
    Returns kept, dropped, trimmed counts plus drop reasons.
    """
    kept = 0
    dropped = 0
    trimmed = 0
    reasons = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(split_ds, "with_format"):
        split_ds = split_ds.with_format("python")

    with path.open("w", encoding="utf-8") as f:
        for record in split_ds:
            sample = {key: record[key] for key in record}
            sample = copy.deepcopy(sample)
            processed = sample
            meta = None
            if processor is not None:
                processed, meta = processor.process(sample)
                if processed is None:
                    dropped += 1
                    reason = meta.get("reason") if meta else None
                    if reason:
                        reasons[reason] += 1
                    continue
                if meta and meta.get("trimmed"):
                    trimmed += 1
            kept += 1
            f.write(json.dumps(processed, ensure_ascii=False) + "\n")

    return kept, dropped, trimmed, reasons


def split_dataset_into_shards(dataset, num_shards: int, seed: int):
    if num_shards == 1:
        return [dataset]
    dataset = dataset.shuffle(seed=seed)
    total = len(dataset)
    base = total // num_shards
    remainder = total % num_shards
    sizes = [base + (1 if idx < remainder else 0) for idx in range(num_shards)]
    shards = []
    offset = 0
    for size in sizes:
        if size == 0:
            shard = dataset.select([])
        else:
            shard = dataset.select(range(offset, offset + size))
        shards.append(shard)
        offset += size
    return shards


def prepare_clients(dataset,
                    client_groups: Sequence[Dict],
                    args,
                    stats_recorder: StatsRecorder,
                    processor: Optional[SampleProcessor]):
    output_root = Path(args.output_dir)
    manifest_clients: List[Dict] = []
    for group_idx, group in enumerate(client_groups):
        labels = group.get("labels") or group.get("families") or group.get("sources")
        if not labels:
            raise ValueError(f"Invalid client group: {group}")
        filter_key = group.get("filter_key", "task_family")
        if filter_key == "task_family":
            unknown = [lab for lab in labels if lab not in FAMILY_CHOICES]
            if unknown:
                raise ValueError(
                    f"Client group refers to unsupported families: {unknown}")
        names = group.get("names") or []
        if not names:
            raise ValueError(f"Client group missing names: {group}")

        subset = dataset.filter(lambda ex: ex[filter_key] in labels)
        if len(subset) < len(names):
            raise ValueError(
                f"Requested {len(names)} shards for {filter_key} {sorted(labels)}, "
                f"but only {len(subset)} samples are available.")
        sorted_labels = sorted(labels)
        shards = split_dataset_into_shards(subset, len(names),
                                           seed=args.seed + group_idx)

        for shard_idx, (name, shard) in enumerate(zip(names, shards)):
            shard_seed = args.seed + group_idx * 1000 + shard_idx
            shard = downsample(shard, args.max_examples_per_client, shard_seed)
            stats_recorder.register_client(name, sorted_labels, len(shard))
            train_ds, val_ds = train_val_split(shard, args.val_frac, shard_seed)
            if len(train_ds) == 0:
                print(f"[WARN] Skipping client {name}: no samples for {filter_key} {labels}")
                continue
            client_dir = output_root / name
            train_path = client_dir / "train.jsonl"
            val_path = client_dir / "val.jsonl"
            train_kept, train_dropped, train_trimmed, train_reasons = \
                process_and_save_split(train_ds, train_path, processor)
            val_kept, val_dropped, val_trimmed, val_reasons = \
                process_and_save_split(val_ds, val_path, processor)

            stats_recorder.record_split(name, "train", len(train_ds), train_kept,
                                        train_dropped, train_trimmed,
                                        train_reasons)
            stats_recorder.record_split(name, "val", len(val_ds), val_kept,
                                        val_dropped, val_trimmed, val_reasons)

            if train_kept == 0:
                print(f"[WARN] Skipping client {name}: no training data after filtering")
                if train_path.exists():
                    train_path.unlink()
                if val_path.exists():
                    val_path.unlink()
                continue
            stats_recorder.accumulate_family_counts(sorted_labels,
                                                    train_kept + val_kept)
            manifest_entry = {
                "name": name,
                "families": sorted_labels,
                "group_key": filter_key,
                "group_values": sorted_labels,
                "train_examples": train_kept,
                "val_examples": val_kept,
                "train_file": relative_path(train_path, output_root),
                "val_file": relative_path(val_path, output_root),
            }
            if filter_key == "source" and len(sorted_labels) == 1:
                manifest_entry["source"] = sorted_labels[0]
            manifest_clients.append(manifest_entry)
    return manifest_clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",
                        default="allenai/tulu-3-sft-mixture",
                        help="HF dataset identifier to download")
    parser.add_argument("--split", default="train",
                        help="Dataset split to process")
    parser.add_argument("--output-dir",
                        default="data/tulu3_federated",
                        help="Directory to store federated shards")
    parser.add_argument("--client-config",
                        type=str,
                        default=None,
                        help="JSON or YAML file describing clients")
    parser.add_argument("--group-by",
                        type=str,
                        default="family",
                        choices=["family", "source"],
                        help="Group clients by task family or source dataset.")
    parser.add_argument("--cache-dir",
                        type=str,
                        default=None,
                        help="Optional datasets cache directory")
    parser.add_argument("--num-proc",
                        type=int,
                        default=4,
                        help="Parallel workers for datasets.map calls")
    parser.add_argument("--val-frac",
                        type=float,
                        default=0.01,
                        help="Validation fraction per client")
    parser.add_argument("--max-samples",
                        type=int,
                        default=None,
                        help="Optional cap on total samples (debugging)")
    parser.add_argument("--max-examples-per-client",
                        type=int,
                        default=None,
                        help="Optional cap for each client subset")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Deterministic shuffle seed")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite existing output directory")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="Pass trust_remote_code=True to load_dataset")
    parser.add_argument("--enable-length-filter",
                        action="store_true",
                        help="Drop/trim samples whose assistant labels disappear "
                             "after truncation.")
    parser.add_argument("--filter-tokenizer-name",
                        type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer used to measure assistant token counts.")
    parser.add_argument("--filter-tokenizer-cache",
                        type=str,
                        default=None,
                        help="Cache directory for the filtering tokenizer.")
    parser.add_argument("--filter-max-length",
                        type=int,
                        default=2048,
                        help="Context window (tokens) used when filtering.")
    parser.add_argument("--filter-min-assistant-tokens",
                        type=int,
                        default=1,
                        help="Minimum assistant tokens required to keep a sample.")
    parser.add_argument("--filter-trim-user-overflow",
                        action="store_true",
                        help="Trim user turns before dropping long samples.")
    parser.add_argument("--filter-user-tail-tokens",
                        type=int,
                        default=1024,
                        help="If trimming user turns, keep only this many tokens "
                             "from the end of each user message.")
    parser.add_argument("--filter-trim-overflow",
                        action="store_true",
                        help="Trim the final assistant turn instead of dropping "
                             "long samples outright.")
    parser.add_argument("--filter-assistant-tail-tokens",
                        type=int,
                        default=1024,
                        help="If trimming, keep only this many tokens from the "
                             "end of the assistant reply.")
    parser.add_argument("--stats-file",
                        type=str,
                        default=None,
                        help="Optional path to write preprocessing statistics.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    ensure_output_dir(output_root, args.overwrite)

    print(f"Loading {args.dataset}:{args.split} ...")
    dataset = load_dataset(args.dataset,
                           split=args.split,
                           cache_dir=args.cache_dir,
                           trust_remote_code=args.trust_remote_code)
    if args.max_samples:
        limit = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(limit))

    sources = sorted(set(dataset["source"]))
    if args.group_by == "family":
        missing = [src for src in sources if src not in SOURCE_TO_FAMILY]
        if missing:
            raise KeyError(
                f"Missing source->family mapping for: {missing}. Update"
                " SOURCE_TO_FAMILY before rerunning.")

        dataset = dataset.map(annotate_task_family,
                              num_proc=args.num_proc,
                              desc="Annotating task families")
        counts = Counter(dataset["task_family"])
        print("Family counts:")
        for fam, cnt in counts.items():
            print(f"  {fam}: {cnt}")

        config_path = Path(args.client_config) if args.client_config else None
        client_cfg = load_client_config(config_path)
        client_groups = normalize_client_specs(client_cfg["clients"], "task_family")
    else:
        counts = Counter(dataset["source"])
        print("Source counts:")
        for src, cnt in counts.items():
            print(f"  {src}: {cnt}")
        if args.client_config:
            config_path = Path(args.client_config)
            client_cfg = load_client_config(config_path)
            client_groups = normalize_client_specs(client_cfg["clients"], "source")
        else:
            client_groups = build_source_groups(sources)

    stats_path = Path(args.stats_file) if args.stats_file else output_root / "stats.txt"
    stats_recorder = StatsRecorder(counts, stats_path)
    processor = build_sample_processor(args)

    manifest_clients = prepare_clients(dataset, client_groups, args,
                                       stats_recorder, processor)
    if not manifest_clients:
        raise RuntimeError("No client splits were produced. Check config.")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "val_fraction": args.val_frac,
        "group_by": args.group_by,
        "source_to_family": SOURCE_TO_FAMILY,
        "clients": manifest_clients,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {manifest_path}")
    stats_recorder.write()
    print(f"Wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
