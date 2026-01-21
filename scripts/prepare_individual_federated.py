#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read the mapping file.")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload or {}


def _pick(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _estimate_length(tokenizer, messages: List[Dict[str, str]],
                     max_len: int) -> int:
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


def _format_gsm8k(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    question = row.get("question")
    answer = row.get("answer")
    if not question or not answer:
        return None
    return [
        {"role": "user", "content": str(question).strip()},
        {"role": "assistant", "content": str(answer).strip()},
    ]


def _format_hellaswag(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    ctx = row.get("ctx")
    if ctx is None:
        ctx_a = row.get("ctx_a", "")
        ctx_b = row.get("ctx_b", "")
        ctx = f"{ctx_a} {ctx_b}".strip()
    endings = row.get("endings")
    label = row.get("label")
    if not ctx or not endings or label is None:
        return None
    try:
        label_idx = int(label)
    except (TypeError, ValueError):
        return None
    choices = []
    for idx, ending in enumerate(endings):
        choices.append(f"{idx + 1}) {ending}")
    prompt = f"{ctx}\n\n" + "\n".join(choices) + "\nAnswer:"
    answer = str(label_idx + 1)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def _format_piqa(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    goal = row.get("goal")
    sol1 = row.get("sol1")
    sol2 = row.get("sol2")
    label = row.get("label")
    if not goal or sol1 is None or sol2 is None or label is None:
        return None
    try:
        label_idx = int(label)
    except (TypeError, ValueError):
        return None
    prompt = f"{goal}\n1) {sol1}\n2) {sol2}\nAnswer:"
    answer = str(label_idx + 1)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def _format_xsum(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    document = row.get("document")
    summary = row.get("summary")
    if not document or summary is None:
        return None
    prompt = f"Document:\n{document}\nSummary:"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(summary).strip()},
    ]


def _format_mbpp(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    text = row.get("text")
    code = row.get("code")
    if not text or code is None:
        return None
    return [
        {"role": "user", "content": str(text).strip()},
        {"role": "assistant", "content": str(code).rstrip()},
    ]


def _format_hotpotqa(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    question = row.get("question")
    answer = row.get("answer")
    if not question or answer is None:
        return None
    prompt = f"{question}\nAnswer:"
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(answer).strip()},
    ]


FORMATTERS = {
    "gsm8k": _format_gsm8k,
    "hellaswag": _format_hellaswag,
    "piqa": _format_piqa,
    "xsum": _format_xsum,
    "mbpp": _format_mbpp,
    "hotpotqa": _format_hotpotqa,
}


def _load_split(dataset: str,
                split: str,
                config: Optional[str],
                streaming: bool,
                cache_dir: Optional[str]):
    if config:
        return load_dataset(dataset,
                            config,
                            split=split,
                            streaming=streaming,
                            cache_dir=cache_dir)
    return load_dataset(dataset,
                        split=split,
                        streaming=streaming,
                        cache_dir=cache_dir)


def _write_record(handle, record: Dict[str, Any]):
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")


def _process_streaming_split(dataset_iter: Iterable[Dict[str, Any]],
                             formatter_name: str,
                             dataset_id: str,
                             tokenizer,
                             max_length: int,
                             drop_long: bool,
                             max_total_samples: Optional[int],
                             max_samples_per_split: Optional[int],
                             val_fraction: Optional[float],
                             rng: random.Random,
                             train_handle,
                             val_handle) -> Tuple[int, int]:
    train_count = 0
    val_count = 0
    total_written = 0
    formatter = FORMATTERS[formatter_name]

    for row in dataset_iter:
        if max_total_samples is not None and total_written >= max_total_samples:
            break
        messages = formatter(row)
        if not messages:
            continue
        if max_length and drop_long:
            length = _estimate_length(tokenizer, messages, max_length)
            if length > max_length:
                continue
        record = {
            "messages": messages,
            "source": dataset_id,
            "dataset": dataset_id,
        }
        is_val = False
        if val_fraction is not None and val_fraction > 0:
            is_val = rng.random() < val_fraction
        if is_val:
            if max_samples_per_split is not None and \
                    val_count >= max_samples_per_split:
                continue
            _write_record(val_handle, record)
            val_count += 1
        else:
            if max_samples_per_split is not None and \
                    train_count >= max_samples_per_split:
                continue
            _write_record(train_handle, record)
            train_count += 1
        total_written += 1
    return train_count, val_count


def _process_iterable(dataset_iter: Iterable[Dict[str, Any]],
                      formatter_name: str,
                      dataset_id: str,
                      tokenizer,
                      max_length: int,
                      drop_long: bool,
                      max_total_samples: Optional[int],
                      max_samples_per_split: Optional[int],
                      split_name: str,
                      handle) -> int:
    count = 0
    total_written = 0
    formatter = FORMATTERS[formatter_name]
    for row in dataset_iter:
        if max_total_samples is not None and total_written >= max_total_samples:
            break
        messages = formatter(row)
        if not messages:
            continue
        if max_length and drop_long:
            length = _estimate_length(tokenizer, messages, max_length)
            if length > max_length:
                continue
        record = {
            "messages": messages,
            "source": dataset_id,
            "dataset": dataset_id,
            "split": split_name,
        }
        _write_record(handle, record)
        count += 1
        total_written += 1
        if max_samples_per_split is not None and count >= max_samples_per_split:
            break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Client mapping YAML file.")
    parser.add_argument("--output-dir", default="data/individual_federated",
                        help="Directory to write clients + manifest.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer",
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer for length filtering.")
    parser.add_argument("--hf-cache-dir",
                        default=os.environ.get("HF_HOME"),
                        help="Optional HF cache directory.")
    parser.add_argument("--streaming",
                        action="store_true",
                        default=None)
    parser.add_argument("--no-streaming",
                        action="store_false",
                        dest="streaming")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap applied to total and per-split samples.")
    parser.add_argument("--max-total-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-client", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--drop-long", action="store_true", default=None)
    parser.add_argument("--no-drop-long", action="store_false", dest="drop_long")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_yaml(config_path)
    defaults = config.get("defaults", {})
    clients = config.get("clients", [])
    if not clients:
        raise ValueError(f"No clients defined in {config_path}")

    output_root = Path(args.output_dir)
    if args.overwrite and output_root.exists():
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer_name = _pick(args.tokenizer, defaults.get("tokenizer"))
    max_length_default = _pick(args.max_length, defaults.get("max_length"), 2048)
    streaming_default = _pick(args.streaming, defaults.get("streaming"), True)
    drop_long_default = _pick(args.drop_long, defaults.get("drop_long"), True)
    val_fraction_default = _pick(args.val_fraction, defaults.get("val_fraction"), 0.01)
    max_total_default = _pick(args.max_total_samples, defaults.get("max_total_samples"))
    max_samples_default = _pick(args.max_samples_per_client,
                                defaults.get("max_samples_per_client"))
    if args.max_samples is not None:
        if max_total_default is None:
            max_total_default = args.max_samples
        if max_samples_default is None:
            max_samples_default = args.max_samples

    tokenizer = None
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length=max_length_default,
            use_fast=False,
            cache_dir=args.hf_cache_dir,
        )

    manifest_clients = []
    for idx, client in enumerate(clients):
        name = client.get("name")
        dataset_id = client.get("dataset")
        split = client.get("split", "train")
        if not name or not dataset_id:
            raise ValueError(f"Invalid client entry: {client}")
        formatter_name = client.get("formatter")
        if formatter_name not in FORMATTERS:
            raise ValueError(
                f"Unknown formatter '{formatter_name}' for client {name}")

        client_max_length = _pick(client.get("max_length"), max_length_default)
        client_streaming = _pick(client.get("streaming"), streaming_default)
        client_drop_long = _pick(client.get("drop_long"), drop_long_default)
        client_val_fraction = _pick(client.get("val_fraction"), val_fraction_default)
        client_max_total = _pick(client.get("max_total_samples"), max_total_default)
        client_max_samples = _pick(client.get("max_samples_per_client"), max_samples_default)

        config_name = client.get("config")
        val_split = client.get("val_split")

        train_ds = _load_split(dataset_id,
                               split,
                               config_name,
                               client_streaming,
                               args.hf_cache_dir)
        val_ds = None
        if val_split:
            try:
                val_ds = _load_split(dataset_id,
                                     val_split,
                                     config_name,
                                     client_streaming,
                                     args.hf_cache_dir)
            except Exception as exc:
                print(f"[WARN] Failed to load val split '{val_split}' for "
                      f"{dataset_id}: {exc}. Falling back to val_fraction.")
                val_ds = None

        client_dir = output_root / name
        client_dir.mkdir(parents=True, exist_ok=True)
        train_path = client_dir / "train.jsonl"
        val_path = client_dir / "val.jsonl"

        rng = random.Random(args.seed + idx)
        with train_path.open("w", encoding="utf-8") as train_f, \
                val_path.open("w", encoding="utf-8") as val_f:
            if val_ds is None:
                train_count, val_count = _process_streaming_split(
                    train_ds,
                    formatter_name,
                    dataset_id,
                    tokenizer,
                    client_max_length,
                    client_drop_long,
                    client_max_total,
                    client_max_samples,
                    client_val_fraction,
                    rng,
                    train_f,
                    val_f,
                )
            else:
                train_count = _process_iterable(
                    train_ds,
                    formatter_name,
                    dataset_id,
                    tokenizer,
                    client_max_length,
                    client_drop_long,
                    client_max_total,
                    client_max_samples,
                    "train",
                    train_f,
                )
                remaining_total = None
                if client_max_total is not None:
                    remaining_total = max(client_max_total - train_count, 0)
                val_count = _process_iterable(
                    val_ds,
                    formatter_name,
                    dataset_id,
                    tokenizer,
                    client_max_length,
                    client_drop_long,
                    remaining_total,
                    client_max_samples,
                    "val",
                    val_f,
                )

        if train_count == 0:
            raise ValueError(
                f"Client {name} produced 0 training samples. Check filters, "
                f"dataset ({dataset_id}), and max-length settings.")

        manifest_clients.append({
            "name": name,
            "group_key": "dataset",
            "group_values": [dataset_id],
            "train_examples": train_count,
            "val_examples": val_count,
            "train_file": f"{name}/train.jsonl",
            "val_file": f"{name}/val.jsonl",
            "source": dataset_id,
        })

    manifest = {
        "dataset": "individual_federated",
        "group_by": "dataset",
        "clients": manifest_clients,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(manifest_clients)} clients to {output_root}")


if __name__ == "__main__":
    main()
