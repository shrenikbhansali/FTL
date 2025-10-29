#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
EVAL_RESULT_DIR = ROOT / "eval_result"
RESULT_ROOT = ROOT / "results"

GLOBAL_CKPTS: Dict[str, str] = {}
CLIENT_PREFIX: Dict[str, str] = {}
BASELINE_PREFIX: Dict[str, str] = {}
LOG_PREFIXES: Dict[str, str] = {}


def configure_run(run_type: str) -> None:
    """Configure checkpoint paths and log prefixes based on run scale."""
    global RESULT_ROOT, GLOBAL_CKPTS, CLIENT_PREFIX, BASELINE_PREFIX, LOG_PREFIXES

    if run_type == "full":
        RESULT_ROOT = ROOT / "results_full"
        GLOBAL_CKPTS = {
            "llama2": "ckpts/full/llama2_composite_meta3.ckpt",
            "qwen_moe": "ckpts/full/qwen15moe_composite_meta3.ckpt",
        }
        CLIENT_PREFIX = {
            "llama2": "ckpts/full/llama2_composite_meta3",
            "qwen_moe": "ckpts/full/qwen15moe_composite_meta3",
        }
        BASELINE_PREFIX = {
            "llama2": "ckpts/full/baselines",
            "qwen_moe": "ckpts/full/baselines",
        }
        LOG_PREFIXES = {
            "global": "fsllm-eval-global-full",
            "clients": "fsllm-eval-clients-full",
            "baselines": "fsllm-eval-baselines-full",
        }
    else:
        RESULT_ROOT = ROOT / "results"
        GLOBAL_CKPTS = {
            "llama2": "ckpts/llama2_composite_meta3.ckpt",
            "qwen_moe": "ckpts/qwen15moe_composite_meta3.ckpt",
        }
        CLIENT_PREFIX = {
            "llama2": "ckpts/llama2_composite_meta3",
            "qwen_moe": "ckpts/qwen15moe_composite_meta3",
        }
        BASELINE_PREFIX = {
            "llama2": "ckpts/baselines",
            "qwen_moe": "ckpts/baselines",
        }
        LOG_PREFIXES = {
            "global": "fsllm-eval-global",
            "clients": "fsllm-eval-clients",
            "baselines": "fsllm-eval-baselines",
        }

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalKey:
    group: str         # global | clients | baselines
    model: str         # llama2 | qwen_moe | ...
    variant: str       # global | client_1 | code_local | ...
    task: str          # mmlu | gsm8k | code


GLOBAL_INDEX = {
    0: EvalKey("global", "llama2", "global", "mmlu"),
    1: EvalKey("global", "llama2", "global", "gsm8k"),
    2: EvalKey("global", "llama2", "global", "code"),
    3: EvalKey("global", "qwen_moe", "global", "mmlu"),
    4: EvalKey("global", "qwen_moe", "global", "gsm8k"),
    5: EvalKey("global", "qwen_moe", "global", "code"),
}

CLIENT_MODELS = [
    "llama2", "llama2", "llama2",
    "llama2", "llama2", "llama2",
    "llama2", "llama2", "llama2",
    "qwen_moe", "qwen_moe", "qwen_moe",
    "qwen_moe", "qwen_moe", "qwen_moe",
    "qwen_moe", "qwen_moe", "qwen_moe",
]
CLIENT_IDS = [
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
]
CLIENT_TASKS = [
    "mmlu", "gsm8k", "code",
    "mmlu", "gsm8k", "code",
    "mmlu", "gsm8k", "code",
    "mmlu", "gsm8k", "code",
    "mmlu", "gsm8k", "code",
    "mmlu", "gsm8k", "code",
]

CLIENT_INDEX = {
    idx: EvalKey(
        "clients",
        CLIENT_MODELS[idx],
        f"client_{CLIENT_IDS[idx]}",
        CLIENT_TASKS[idx],
    )
    for idx in range(len(CLIENT_MODELS))
}

BASE_MODELS = ["llama2", "llama2", "llama2", "llama2",
               "qwen_moe", "qwen_moe", "qwen_moe", "qwen_moe"]
BASE_VARIANTS = [
    "code_local", "gsm8k_local", "instr_local", "all_in_one",
    "code_local", "gsm8k_local", "instr_local", "all_in_one",
]
BASE_TASKS = ["mmlu", "gsm8k", "code"]

BASE_INDEX: Dict[int, EvalKey] = {}
for idx in range(24):
    mid = idx // 3
    tid = idx % 3
    BASE_INDEX[idx] = EvalKey(
        "baselines",
        BASE_MODELS[mid],
        BASE_VARIANTS[mid],
        BASE_TASKS[tid],
    )

def client_ckpt(model: str, client_id: str) -> str:
    prefix = CLIENT_PREFIX[model]
    return f"{prefix}/clients/{client_id}/{client_id}.ckpt"


def baseline_ckpt(model: str, variant: str) -> str:
    root = BASELINE_PREFIX[model]
    return f"{root}/{model}_{variant}.ckpt"

# ---------------------------------------------------------------------------
# Helpers for parsing logs
# ---------------------------------------------------------------------------


def load_text_pair(base: Path) -> str:
    """Concatenate .out and .err contents (when present)."""
    text = ""
    if base.exists():
        text += base.read_text(errors="ignore")
    err_path = base.with_suffix(".err")
    if err_path.exists():
        text += "\n" + err_path.read_text(errors="ignore")
    return text


def parse_gsm8k(text: str) -> Optional[float]:
    matches = re.findall(r"correct rate:\s*([0-9]*\.?[0-9]+)", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def parse_code(text: str) -> Optional[float]:
    matches = re.findall(r"Average accuracy[:\s]*([0-9]*\.?[0-9]+)", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def detect_error(text: str) -> Optional[str]:
    error_markers = [
        "Traceback (most recent call last):",
        "FileNotFoundError",
        "RuntimeError",
        "ValueError",
        "KeyError",
    ]
    for marker in error_markers:
        if marker in text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            tail = " | ".join(lines[-5:])
            return marker if not tail else tail[-200:]
    return None


def collect_log_metrics(prefix: str,
                        index_map: Dict[int, EvalKey]) -> Dict[EvalKey, Tuple[Optional[float], str, Optional[str]]]:
    results: Dict[EvalKey, List[Tuple[float, Optional[float], str, Optional[str]]]] = {}
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_([0-9]+)\.out$")

    for path in LOG_DIR.glob(f"{prefix}_*.out"):
        match = pattern.match(path.name)
        if not match:
            continue
        array_idx = int(match.group(2))
        if array_idx not in index_map:
            continue
        key = index_map[array_idx]
        text = load_text_pair(path)
        error_msg = detect_error(text)

        metric: Optional[float] = None
        if key.task == "gsm8k":
            metric = parse_gsm8k(text)
        elif key.task == "code":
            metric = parse_code(text)

        timestamp = path.stat().st_mtime
        results.setdefault(key, []).append(
            (timestamp, metric, str(path.relative_to(ROOT)), error_msg)
        )

    final: Dict[EvalKey, Tuple[Optional[float], str, Optional[str]]] = {}
    for key, entries in results.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        chosen = None
        for entry in entries:
            _, metric, source, error_msg = entry
            if metric is not None:
                chosen = (metric, source, None)
                break
        if chosen is None:
            # take latest entry even if metric missing to surface errors
            _, metric, source, error_msg = entries[0]
            chosen = (metric, source, error_msg)
        final[key] = chosen
    return final

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_mmlu_metrics() -> Dict[EvalKey, Tuple[Optional[float], str]]:
    metrics: Dict[EvalKey, Tuple[Optional[float], str]] = {}

    for key in GLOBAL_INDEX.values():
        if key.task != "mmlu":
            continue
        save_to = GLOBAL_CKPTS[key.model]
        json_path = EVAL_RESULT_DIR / f"accuracies_{save_to.replace('/', '_')}.json"
        metric = _read_mmlu_json(json_path)
        metrics[key] = (metric, str(json_path.relative_to(ROOT)))

    for key in CLIENT_INDEX.values():
        if key.task != "mmlu":
            continue
        ckpt = client_ckpt(key.model, key.variant)
        json_path = EVAL_RESULT_DIR / f"accuracies_{ckpt.replace('/', '_')}.json"
        metric = _read_mmlu_json(json_path)
        metrics[key] = (metric, str(json_path.relative_to(ROOT)))

    for key in BASE_INDEX.values():
        if key.task != "mmlu":
            continue
        ckpt = baseline_ckpt(key.model, key.variant)
        json_path = EVAL_RESULT_DIR / f"accuracies_{ckpt.replace('/', '_')}.json"
        metric = _read_mmlu_json(json_path)
        metrics[key] = (metric, str(json_path.relative_to(ROOT)))

    return metrics


def _read_mmlu_json(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data.get("weighted_accuracy"))
    except (ValueError, json.JSONDecodeError, TypeError):
        return None


def build_expected_entries() -> List[EvalKey]:
    seen = set()
    entries: List[EvalKey] = []
    for index_map in (GLOBAL_INDEX, CLIENT_INDEX, BASE_INDEX):
        for key in index_map.values():
            if key not in seen:
                seen.add(key)
                entries.append(key)
    entries.sort(key=lambda k: (k.group, k.model, k.variant, k.task))
    return entries


def aggregate() -> Tuple[List[Tuple[EvalKey, Optional[float], str, Optional[str]]], List[EvalKey]]:
    mmlu_metrics = load_mmlu_metrics()
    global_log_metrics = collect_log_metrics(LOG_PREFIXES["global"], GLOBAL_INDEX)
    client_log_metrics = collect_log_metrics(LOG_PREFIXES["clients"], CLIENT_INDEX)
    base_log_metrics = collect_log_metrics(LOG_PREFIXES["baselines"], BASE_INDEX)

    rows: List[Tuple[EvalKey, Optional[float], str, Optional[str]]] = []
    missing: List[EvalKey] = []

    for key in build_expected_entries():
        metric = None
        source = ""
        note = None

        if key.task == "mmlu":
            metric, source = mmlu_metrics.get(key, (None, str(EVAL_RESULT_DIR.relative_to(ROOT))))
            if source is None:
                source = str(EVAL_RESULT_DIR.relative_to(ROOT))
        else:
            if key.group == "global":
                metric_info = global_log_metrics.get(key)
            elif key.group == "clients":
                metric_info = client_log_metrics.get(key)
            else:
                metric_info = base_log_metrics.get(key)
            if metric_info:
                metric, source, note = metric_info

        if metric is None:
            missing.append(key)
        rows.append((key, metric, source, note))
    return rows, missing

# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def format_table(rows: List[Tuple[EvalKey, Optional[float], str, Optional[str]]]) -> str:
    headers = ("Group", "Model", "Variant", "Task", "Metric", "Status/Notes", "Source")
    formatted_rows = []

    for key, metric, source, note in rows:
        if metric is None:
            metric_str = "--"
            status = "missing" if note is None else f"error ({note})"
        else:
            metric_str = f"{metric * 100:.2f}%"
            status = "ok"
        formatted_rows.append((
            key.group,
            key.model,
            key.variant,
            key.task,
            metric_str,
            status,
            source,
        ))

    widths = [len(h) for h in headers]
    for row in formatted_rows:
        widths = [max(w, len(str(cell))) for w, cell in zip(widths, row)]

    def fmt_row(row: Tuple[str, ...]) -> str:
        return " | ".join(
            str(cell).ljust(width)
            for cell, width in zip(row, widths)
        )

    lines = [fmt_row(headers), fmt_row(tuple("-" * w for w in widths))]
    for row in formatted_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize evaluation outputs and flag missing results.",
    )
    parser.add_argument(
        "--run-type",
        choices=["smoke", "full"],
        default="smoke",
        help="Which pipeline run to summarize (smoke=default, full=scaled runs)",
    )
    parser.add_argument(
        "--model",
        choices=["both", "llama2", "qwen_moe"],
        default="both",
        help="Filter results by model (default shows both LLaMA and Qwen)",
    )
    args = parser.parse_args()

    configure_run(args.run_type)

    rows, missing = aggregate()

    if args.model != "both":
        rows = [row for row in rows if row[0].model == args.model]
        missing = [key for key in missing if key.model == args.model]
    print(format_table(rows))
    print()
    if missing:
        print(f"{len(missing)} evaluation entries are missing metrics.")
        for key in missing:
            print(f"  - {key.group}/{key.model}/{key.variant}/{key.task}")
    else:
        print("All expected evaluation entries have recorded metrics.")


if __name__ == "__main__":
    main()
