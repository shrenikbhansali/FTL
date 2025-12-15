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

The produced directory contains one sub-directory per client with
``train.jsonl``/``val.jsonl`` files and a manifest describing the
configuration.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset

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


def save_split(dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(path, orient="records", lines=True)


def prepare_clients(dataset, client_specs: Sequence[Dict], args):
    output_root = Path(args.output_dir)
    manifest_clients: List[Dict] = []
    for spec in client_specs:
        name = spec.get("name")
        families = set(spec.get("families", []))
        if not name or not families:
            raise ValueError(f"Invalid client spec: {spec}")
        unknown = [fam for fam in families if fam not in FAMILY_CHOICES]
        if unknown:
            raise ValueError(
                f"Client {name} refers to unsupported families: {unknown}")
        subset = dataset.filter(lambda ex: ex["task_family"] in families)
        subset = downsample(subset, args.max_examples_per_client, args.seed)
        train_ds, val_ds = train_val_split(subset, args.val_frac, args.seed)
        if len(train_ds) == 0:
            print(f"[WARN] Skipping client {name}: no samples for families {families}")
            continue
        client_dir = output_root / name
        train_path = client_dir / "train.jsonl"
        val_path = client_dir / "val.jsonl"
        save_split(train_ds, train_path)
        save_split(val_ds, val_path)
        manifest_clients.append({
            "name": name,
            "families": sorted(families),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "train_file": relative_path(train_path, output_root),
            "val_file": relative_path(val_path, output_root),
        })
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
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    ensure_output_dir(output_root, args.overwrite)

    config_path = Path(args.client_config) if args.client_config else None
    client_cfg = load_client_config(config_path)

    print(f"Loading {args.dataset}:{args.split} ...")
    dataset = load_dataset(args.dataset,
                           split=args.split,
                           cache_dir=args.cache_dir,
                           trust_remote_code=args.trust_remote_code)
    if args.max_samples:
        limit = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(limit))

    sources = sorted(set(dataset["source"]))
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

    manifest_clients = prepare_clients(dataset, client_cfg["clients"], args)
    if not manifest_clients:
        raise RuntimeError("No client splits were produced. Check config.")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "val_fraction": args.val_frac,
        "source_to_family": SOURCE_TO_FAMILY,
        "clients": manifest_clients,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
