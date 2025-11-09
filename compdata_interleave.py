#!/usr/bin/env python3
"""Build an interleaved version of the composite dataset used by FS-LLM."""

import argparse
import json
import os
import random

from compdata import load_chat_dolly, load_code_alpaca, load_gsm8k


def ok(record):
    """Drop malformed samples that have neither prompt nor response."""
    return (record["instruction"] or record["input"]) and record["output"]


def interleave_round_robin(sources, order):
    """Round-robin interleave datasets according to the requested order."""
    missing = [name for name in order if name not in sources]
    if missing:
        raise ValueError(f"Unknown datasets in pattern: {missing}")
    progress = {name: 0 for name in sources}
    total = sum(len(samples) for samples in sources.values())
    interleaved = []
    while len(interleaved) < total:
        appended = False
        for name in order:
            idx = progress[name]
            samples = sources[name]
            if idx < len(samples):
                interleaved.append(samples[idx])
                progress[name] += 1
                appended = True
        if not appended:
            # No dataset had remaining samples; exit early.
            break
    return interleaved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interleaved_llm3.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--max_chat", type=int, default=None)
    ap.add_argument("--max_code", type=int, default=None)
    ap.add_argument("--max_math", type=int, default=None)
    ap.add_argument(
        "--pattern",
        nargs="+",
        default=["code", "chat", "math"],
        help="Round-robin order; options: chat, code, math.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    chat = [r for r in load_chat_dolly(args.max_chat, args.seed) if ok(r)]
    code = [r for r in load_code_alpaca(args.max_code, args.seed) if ok(r)]
    math = [r for r in load_gsm8k(args.max_math, args.seed) if ok(r)]

    random.seed(args.seed)
    for subset in (chat, code, math):
        random.shuffle(subset)

    per_category_paths = {
        "chat": os.path.join(args.out_dir, "chat_dolly.json"),
        "code": os.path.join(args.out_dir, "code_alpaca.json"),
        "math": os.path.join(args.out_dir, "math_gsm8k.json"),
    }
    for name, samples in (("chat", chat), ("code", code), ("math", math)):
        with open(per_category_paths[name], "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)

    sources = {"chat": chat, "code": code, "math": math}
    composite = interleave_round_robin(sources, args.pattern)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(composite, f, ensure_ascii=False)

    print(f"Wrote {len(chat)} chat, {len(code)} code, {len(math)} math")
    print(f"Interleaved total: {len(composite)} -> {args.out}")


if __name__ == "__main__":
    main()
