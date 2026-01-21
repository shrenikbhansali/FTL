import json
import os

import transformers
from datasets import load_dataset
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)


def _tokenize(text: str):
    return [tok for tok in text.lower().split() if tok]


def _lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, tok_a in enumerate(a, 1):
        row = dp[i]
        prev = dp[i - 1]
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = max(prev[j], row[j - 1])
    return dp[-1][-1]


def _rouge_l_f1(pred: str, ref: str):
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0, 0.0, 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    precision = lcs / float(len(pred_toks))
    recall = lcs / float(len(ref_toks))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _get_max_samples(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_samples"):
        return int(cfg.eval.max_samples)
    return None


def _get_max_new_tokens(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_new_tokens"):
        return int(cfg.eval.max_new_tokens)
    return 128


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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__xsum.json")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)

    max_samples = _get_max_samples(init_cfg)
    max_new_tokens = _get_max_new_tokens(init_cfg)
    generate_kwargs = dict(max_new_tokens=max_new_tokens,
                           do_sample=False,
                           temperature=1.0,
                           top_p=1.0)

    dataset = load_dataset("xsum", split="validation", cache_dir=data_root)

    total = 0
    sum_f1 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0

    for sample in tqdm(dataset, desc="xsum"):
        if max_samples is not None and total >= max_samples:
            break
        document = sample.get("document")
        summary = sample.get("summary")
        if not document or summary is None:
            continue
        prompt = f"Document:\n{document}\nSummary:"
        pred = bot.generate(prompt, generate_kwargs)
        f1, prec, rec = _rouge_l_f1(pred or "", summary)
        sum_f1 += f1
        sum_prec += prec
        sum_rec += rec
        total += 1

    if total:
        rouge_l_f1 = sum_f1 / total
        rouge_l_prec = sum_prec / total
        rouge_l_rec = sum_rec / total
    else:
        rouge_l_f1 = 0.0
        rouge_l_prec = 0.0
        rouge_l_rec = 0.0

    payload = {
        "rougeL_f1": rouge_l_f1,
        "rougeL_precision": rouge_l_prec,
        "rougeL_recall": rouge_l_rec,
        "total_examples": total,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"XSum results written to {out_path}")


if __name__ == "__main__":
    main()
