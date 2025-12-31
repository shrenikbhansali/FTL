#!/usr/bin/env python3
"""Compare GSM8K generations across base Llama and multiple checkpoints."""

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Optional

import torch

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.llm.eval.eval_for_gsm8k.eval import (
    build_prompt,
    clean_answer,
    extract_answer_from_output,
)


def _ensure_gsm8k_data(data_root: str) -> str:
    os.makedirs(data_root, exist_ok=True)
    fp = os.path.join(data_root, "gsm8k_test.jsonl")
    if not os.path.exists(fp):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            data_root,
        )
        os.rename(os.path.join(data_root, "test.jsonl"), fp)
    return fp


def _default_lora_args() -> List[Dict]:
    return [
        {
            "adapter_package": "peft",
            "adapter_method": "lora",
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
        }
    ]


def _build_cfg(
    ckpt_path: str,
    use_adapter: bool,
    fp16: bool,
    data_root: str,
) -> "Config":
    cfg = global_cfg.clone()
    cfg.use_gpu = True
    cfg.device = 0
    cfg.seed = 42
    cfg.data.root = data_root
    cfg.model.type = "meta-llama/Llama-2-7b-hf@huggingface_llm"
    cfg.federate.save_to = ckpt_path
    cfg.llm.tok_len = 2048
    cfg.llm.chat.max_len = 512
    cfg.llm.chat.max_history_len = 0
    cfg.llm.adapter.use = use_adapter
    cfg.llm.adapter.args = _default_lora_args()
    cfg.train.is_enable_half = fp16
    return cfg


def _gather_extra_ckpts(ckpt_dir: str, include_all: bool) -> List[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    entries = []
    for name in os.listdir(ckpt_dir):
        if not name.endswith(".ckpt"):
            continue
        entries.append(os.path.join(ckpt_dir, name))
    if include_all:
        return sorted(entries)
    keywords = ("unlearn", "loo", "bank")
    return sorted([p for p in entries if any(k in os.path.basename(p) for k in keywords)])


def _dedupe(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="ckpts/full")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--include-all", action="store_true",
                        help="Include every .ckpt in ckpt-dir (can be slow).")
    parser.add_argument("--max-questions", type=int, default=2)
    args = parser.parse_args()

    update_logger(global_cfg, clear_before_add=True)
    setup_seed(42)

    gsm8k_path = _ensure_gsm8k_data(args.data_root)
    samples = load_jsonl(gsm8k_path, instruction="question", output="answer")
    samples = samples[: args.max_questions]

    model_specs = [
        {
            "name": "llama2_base",
            "ckpt": os.path.join(args.ckpt_dir, "__base__missing__.ckpt"),
            "use_adapter": False,
            "is_base": True,
        },
    ]

    tulu_ckpts = [
        ("tulu3_federated_centralized", os.path.join(args.ckpt_dir, "tulu3_federated_centralized.ckpt")),
        ("tulu3_federated_fedavg", os.path.join(args.ckpt_dir, "tulu3_federated_fedavg.ckpt")),
        ("tulu3_federated_unlearn_bank_perclient", os.path.join(args.ckpt_dir, "tulu3_federated_unlearn_bank_perclient.ckpt")),
    ]
    for name, path in tulu_ckpts:
        if os.path.exists(path):
            model_specs.append(
                {"name": name, "ckpt": path, "use_adapter": True, "is_base": False}
            )

    extra_ckpts = _gather_extra_ckpts(args.ckpt_dir, args.include_all)
    extra_ckpts = _dedupe(extra_ckpts)
    existing_ckpts = {spec["ckpt"] for spec in model_specs}
    for path in extra_ckpts:
        if path in existing_ckpts:
            continue
        model_specs.append(
            {
                "name": os.path.basename(path).replace(".ckpt", ""),
                "ckpt": path,
                "use_adapter": True,
                "is_base": False,
            }
        )

    results = {
        "fp16": args.fp16,
        "questions": [],
        "models": [spec["name"] for spec in model_specs],
    }

    for idx, sample in enumerate(samples):
        question = sample["instruction"]
        answer = extract_answer_from_output(sample["output"])
        prompt = build_prompt(question, n_shot=8, cot_flag=True)
        results["questions"].append(
            {
                "index": idx,
                "question": question,
                "answer": answer,
                "prompt": prompt,
            }
        )

    for spec in model_specs:
        print(f"\n=== Evaluating {spec['name']} ===")
        cfg = _build_cfg(
            ckpt_path=spec["ckpt"],
            use_adapter=spec["use_adapter"],
            fp16=args.fp16,
            data_root=args.data_root,
        )
        bot = FSChatBot(cfg)
        model_outputs: List[Dict[str, str]] = []
        for q in results["questions"]:
            prompt = q["prompt"]
            try:
                completion = bot.generate(
                    prompt,
                    generate_kwargs=dict(
                        max_new_tokens=256,
                        top_p=0.95,
                        temperature=0.8,
                    ),
                )
                cleaned = clean_answer(completion)
            except Exception as exc:
                completion = f"<error: {exc}>"
                cleaned = "<error>"
            model_outputs.append(
                {
                    "completion": completion,
                    "cleaned_answer": cleaned,
                }
            )
            print(f"Q: {q['question']}")
            print(f"GT: {q['answer']}")
            print(f"Model: {cleaned}")
            print(f"Completion: {completion}\n")

        results.setdefault("outputs", {})[spec["name"]] = model_outputs
        del bot
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    out_dir = os.path.join("results_debug", "gsm8k_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"gsm8k_compare_{int(time.time())}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
