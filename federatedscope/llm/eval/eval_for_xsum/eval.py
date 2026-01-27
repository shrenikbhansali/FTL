import json
import os

import transformers
from datasets import load_dataset
from tqdm import tqdm
try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)


def _get_eval_split(cfg) -> str:
    split = "test"
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "split"):
        split_val = cfg.eval.split
        if isinstance(split_val, str):
            split = split_val
        elif isinstance(split_val, (list, tuple)) and split_val:
            split = split_val[0]
    return split


def _get_rouge_scorer() -> "rouge_scorer.RougeScorer":
    if rouge_scorer is None:
        raise RuntimeError(
            "rouge-score is required for XSum evaluation. "
            "Install with `pip install rouge-score`."
        )
    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )


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

    split = _get_eval_split(init_cfg)
    dataset = load_dataset("xsum", split=split, cache_dir=data_root)
    scorer = _get_rouge_scorer()

    total = 0
    sums = {
        "rouge1": {"f": 0.0, "p": 0.0, "r": 0.0},
        "rouge2": {"f": 0.0, "p": 0.0, "r": 0.0},
        "rougeL": {"f": 0.0, "p": 0.0, "r": 0.0},
    }

    for sample in tqdm(dataset, desc="xsum"):
        if max_samples is not None and total >= max_samples:
            break
        document = sample.get("document")
        summary = sample.get("summary")
        if not document or summary is None:
            continue
        prompt = f"Document:\n{document}\nSummary:"
        pred = bot.generate(prompt, generate_kwargs)
        scores = scorer.score(summary, pred or "")
        for name, metrics in sums.items():
            score = scores.get(name)
            if score is None:
                continue
            metrics["f"] += score.fmeasure
            metrics["p"] += score.precision
            metrics["r"] += score.recall
        total += 1

    payload = {"total_examples": total}
    for name, metrics in sums.items():
        if total:
            f1 = metrics["f"] / total
            prec = metrics["p"] / total
            rec = metrics["r"] / total
        else:
            f1 = prec = rec = 0.0
        payload[f"{name}_f1"] = f1
        payload[f"{name}_precision"] = prec
        payload[f"{name}_recall"] = rec

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"XSum results written to {out_path}")


if __name__ == "__main__":
    main()
