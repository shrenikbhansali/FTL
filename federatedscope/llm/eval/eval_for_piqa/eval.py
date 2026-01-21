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
    match = re.search(r"\b([12])\b", norm)
    if match:
        return int(match.group(1)) - 1
    match = re.search(r"\b([ab])\b", norm)
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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__piqa.json")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)

    max_samples = _get_max_samples(init_cfg)
    max_new_tokens = _get_max_new_tokens(init_cfg)
    generate_kwargs = dict(max_new_tokens=max_new_tokens,
                           do_sample=False,
                           temperature=1.0,
                           top_p=1.0)

    try:
        dataset = load_dataset("piqa", split="validation", cache_dir=data_root)
    except Exception as exc:
        payload = {
            "error": f"Failed to load piqa dataset: {exc}",
            "weighted_accuracy": None,
            "total_examples": 0,
            "categories": {},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(payload["error"])
        print(f"PIQA results written to {out_path}")
        return

    correct = 0
    total = 0
    for sample in tqdm(dataset, desc="piqa"):
        if max_samples is not None and total >= max_samples:
            break
        goal = sample.get("goal", "")
        sol1 = sample.get("sol1")
        sol2 = sample.get("sol2")
        if sol1 is None or sol2 is None:
            continue
        prompt = f"{goal}\n1) {sol1}\n2) {sol2}\nAnswer:"
        pred = bot.generate(prompt, generate_kwargs)
        pred_idx = _extract_choice(pred or "")
        label = sample.get("label")
        try:
            gold_idx = int(label)
        except (TypeError, ValueError):
            continue
        if pred_idx == gold_idx:
            correct += 1
        total += 1

    accuracy = float(correct / total) if total else 0.0
    payload = {
        "weighted_accuracy": accuracy,
        "total_examples": total,
        "categories": {"piqa": accuracy},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"PIQA results written to {out_path}")


if __name__ == "__main__":
    main()
