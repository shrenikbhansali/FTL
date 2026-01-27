import json
import os
import re
import string

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
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _f1_score(pred: str, truth: str) -> float:
    pred_norm = _normalize(pred)
    truth_norm = _normalize(truth)
    if truth_norm in {"yes", "no", "noanswer"}:
        return float(pred_norm == truth_norm)
    pred_tokens = pred_norm.split()
    truth_tokens = truth_norm.split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in truth_tokens:
        count = common.get(tok, 0)
        if count > 0:
            overlap += 1
            common[tok] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, truth: str) -> bool:
    return _normalize(pred) == _normalize(truth)


def _get_max_samples(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_samples"):
        return int(cfg.eval.max_samples)
    return None


def _get_max_new_tokens(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_new_tokens"):
        return int(cfg.eval.max_new_tokens)
    return 32


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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__hotpotqa.json")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)

    max_samples = _get_max_samples(init_cfg)
    max_new_tokens = _get_max_new_tokens(init_cfg)
    generate_kwargs = dict(max_new_tokens=max_new_tokens,
                           do_sample=False,
                           temperature=1.0,
                           top_p=1.0)

    dataset = load_dataset("hotpot_qa",
                           "distractor",
                           split="validation",
                           cache_dir=data_root)

    total = 0
    em_total = 0
    f1_total = 0.0

    for sample in tqdm(dataset, desc="hotpotqa"):
        if max_samples is not None and total >= max_samples:
            break
        question = sample.get("question")
        answer = sample.get("answer")
        if not question or answer is None:
            continue
        context = sample.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        parts = []
        for title, sents in zip(titles, sentences):
            if not sents:
                continue
            text = " ".join([str(s).strip() for s in sents if s])
            if not text:
                continue
            if title:
                parts.append(f"[{str(title).strip()}] {text}")
            else:
                parts.append(text)
        context_block = "\n".join(parts).strip()
        if context_block:
            prompt = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"{question}\nAnswer:"
        pred = bot.generate(prompt, generate_kwargs)
        pred_text = pred or ""
        if _exact_match(pred_text, answer):
            em_total += 1
        f1_total += _f1_score(pred_text, answer)
        total += 1

    em = float(em_total / total) if total else 0.0
    f1 = float(f1_total / total) if total else 0.0
    payload = {
        "em": em,
        "f1": f1,
        "total_examples": total,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"HotPotQA results written to {out_path}")


if __name__ == "__main__":
    main()
