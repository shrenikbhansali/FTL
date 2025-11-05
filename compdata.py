#!/usr/bin/env python3
# tools/build_composite_llm3.py
import argparse
import json
import os
import random
from datasets import load_dataset

def to_record(instruction, inp, output, category, source=None):
    rec = {
        "instruction": instruction.strip() if instruction else "",
        "input": inp.strip() if inp else "",
        "output": output.strip() if output else "",
        "category": category
    }
    if source:
        rec["source"] = source  # handy for debugging; FS-LLM will ignore
    return rec

def load_chat_dolly(max_n=None, seed=42):
    ds = load_dataset("databricks/databricks-dolly-15k")["train"]
    data = []
    for ex in ds:
        # Dolly has instruction/context/response
        # (+ fine-grained task category we ignore)
        data.append(to_record(
            instruction=ex.get("instruction", ""),
            inp=ex.get("context", "") or "",
            output=ex.get("response", ""),
            category="chat",
            source="dolly-15k"
        ))
    if max_n:
        random.seed(seed)
        random.shuffle(data)
        data = data[:max_n]
    return data

def load_code_alpaca(max_n=None, seed=42):
    ds = load_dataset("sahil2801/CodeAlpaca-20k")["train"]
    data = []
    for ex in ds:
        data.append(to_record(
            instruction=ex.get("instruction", ""),
            inp=ex.get("input", "") or "",
            output=ex.get("output", ""),
            category="code",
            source="codealpaca-20k"
        ))
    if max_n:
        random.seed(seed)
        random.shuffle(data)
        data = data[:max_n]
    return data

def load_gsm8k(max_n=None, seed=42):
    # We will use the 'train' split to build SFT-style supervision.
    ds = load_dataset("openai/gsm8k", "main")["train"]
    INSTR = (
        "Solve the grade-school math problem step by step and give the final "
        "answer.")
    data = []
    for ex in ds:
        q = ex.get("question", "")
        a = ex.get("answer", "")
        data.append(to_record(
            instruction=INSTR,
            inp=q,
            output=a,          # keep full rationale; FS-LLM will train on it
            category="math",
            source="gsm8k"
        ))
    if max_n:
        random.seed(seed)
        random.shuffle(data)
        data = data[:max_n]
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/composite_llm3.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument(
        "--max_chat",
        type=int,
        default=None,
        help="optional cap for Dolly-15k")
    ap.add_argument(
        "--max_code",
        type=int,
        default=None,
        help="optional cap for CodeAlpaca-20k")
    ap.add_argument(
        "--max_math",
        type=int,
        default=None,
        help="optional cap for GSM8K")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    chat = load_chat_dolly(max_n=args.max_chat, seed=args.seed)
    code = load_code_alpaca(max_n=args.max_code, seed=args.seed)
    math = load_gsm8k(max_n=args.max_math, seed=args.seed)

    # Basic hygiene: drop empties
    def ok(r):
        return (r["instruction"] or r["input"]) and r["output"]

    chat = [r for r in chat if ok(r)]
    code = [r for r in code if ok(r)]
    math = [r for r in math if ok(r)]

    # Write per-category (optional)
    chat_path = os.path.join(args.out_dir, "chat_dolly.json")
    code_path = os.path.join(args.out_dir, "code_alpaca.json")
    math_path = os.path.join(args.out_dir, "math_gsm8k.json")

    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False)
    with open(code_path, "w", encoding="utf-8") as f:
        json.dump(code, f, ensure_ascii=False)
    with open(math_path, "w", encoding="utf-8") as f:
        json.dump(math, f, ensure_ascii=False)

    # Composite for FS-LLM + MetaSplitter
    composite = chat + code + math
    random.seed(args.seed)
    random.shuffle(composite)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(composite, f, ensure_ascii=False)

    print(f"Wrote {len(chat)} chat, {len(code)} code, {len(math)} math")
    print(f"Composite total: {len(composite)} -> {args.out}")

if __name__ == "__main__":
    main()
