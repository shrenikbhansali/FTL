import json
import os
import re

import transformers
from datasets import load_dataset
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _extract_choice(text: str):
    norm = _normalize(text)
    match = re.search(r"\b([1-4])\b", norm)
    if match:
        return int(match.group(1)) - 1
    match = re.search(r"\b([a-d])\b", norm)
    if match:
        return ord(match.group(1)) - ord("a")
    return None


def _get_max_samples(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_samples"):
        return int(cfg.eval.max_samples)
    return None


def _get_max_new_tokens(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_new_tokens"):
        return int(cfg.eval.max_new_tokens)
    return 8


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    bot = FSChatBot(init_cfg)

    eval_dir = "eval_result"
    if hasattr(init_cfg, "outdir") and init_cfg.outdir:
        eval_dir = os.path.join(init_cfg.outdir, "eval_result")
    os.makedirs(eval_dir, exist_ok=True)
    save_name = init_cfg.federate.save_to.replace("/", "_")
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__hellaswag.json")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)

    max_samples = _get_max_samples(init_cfg)
    max_new_tokens = _get_max_new_tokens(init_cfg)
    generate_kwargs = dict(max_new_tokens=max_new_tokens,
                           do_sample=False,
                           temperature=1.0,
                           top_p=1.0)

    dataset = load_dataset("hellaswag",
                           split="validation",
                           cache_dir=data_root)

    correct = 0
    total = 0
    for sample in tqdm(dataset, desc="hellaswag"):
        if max_samples is not None and total >= max_samples:
            break
        endings = sample.get("endings") or []
        if len(endings) != 4:
            continue
        ctx = sample.get("ctx")
        if ctx is None:
            ctx_a = sample.get("ctx_a", "")
            ctx_b = sample.get("ctx_b", "")
            ctx = f"{ctx_a} {ctx_b}".strip()
        prompt_lines = [ctx, "", "1) " + endings[0], "2) " + endings[1],
                        "3) " + endings[2], "4) " + endings[3], "Answer:"]
        prompt = "\n".join(prompt_lines)
        pred = bot.generate(prompt, generate_kwargs)
        pred_idx = _extract_choice(pred or "")
        gold = sample.get("label")
        try:
            gold_idx = int(gold)
        except (TypeError, ValueError):
            continue
        if pred_idx == gold_idx:
            correct += 1
        total += 1

    accuracy = float(correct / total) if total else 0.0
    payload = {
        "weighted_accuracy": accuracy,
        "total_examples": total,
        "categories": {"hellaswag": accuracy},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"HellaSwag results written to {out_path}")


if __name__ == "__main__":
    main()
