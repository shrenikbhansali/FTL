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

DEFAULT_TASKS = ["boolq", "rte", "cb", "copa", "wic"]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _extract_yes_no(text: str):
    text = _normalize(text)
    if text.startswith("yes") or " yes" in text:
        return "yes"
    if text.startswith("no") or " no" in text:
        return "no"
    return None


def _extract_label(text: str, labels):
    text = _normalize(text)
    for label in labels:
        if label in text:
            return label
    return None


def _extract_choice(text: str):
    match = re.search(r"\b([12])\b", text)
    if match:
        return match.group(1)
    return None


def _get_tasks(cfg):
    tasks = None
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "superglue_tasks"):
        tasks = cfg.eval.superglue_tasks
    if tasks is None:
        return DEFAULT_TASKS
    if isinstance(tasks, str):
        return [t.strip() for t in tasks.split(",") if t.strip()]
    return list(tasks)


def _get_max_samples(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_samples"):
        return int(cfg.eval.max_samples)
    return None


def _evaluate_task(task, dataset, bot, max_samples, generate_kwargs):
    correct = 0
    total = 0
    for sample in tqdm(dataset, desc=task):
        if max_samples is not None and total >= max_samples:
            break
        if task == "boolq":
            prompt = (
                "Passage: {passage}\n"
                "Question: {question}\n"
                "Answer (yes or no):"
            ).format(**sample)
            pred = bot.generate(prompt, generate_kwargs)
            pred_label = _extract_yes_no(pred or "")
            gold = "yes" if sample.get("label") == 1 else "no"
        elif task == "rte":
            prompt = (
                "Premise: {premise}\n"
                "Hypothesis: {hypothesis}\n"
                "Answer with entailment or not_entailment:"
            ).format(**sample)
            pred = bot.generate(prompt, generate_kwargs)
            labels = ["entailment", "not entailment", "not_entailment"]
            pred_label = _extract_label(pred or "", labels)
            if pred_label == "not entailment":
                pred_label = "not_entailment"
            gold = "entailment" if sample.get("label") == 0 else "not_entailment"
        elif task == "cb":
            prompt = (
                "Premise: {premise}\n"
                "Hypothesis: {hypothesis}\n"
                "Answer with entailment, contradiction, or neutral:"
            ).format(**sample)
            pred = bot.generate(prompt, generate_kwargs)
            labels = ["entailment", "contradiction", "neutral"]
            pred_label = _extract_label(pred or "", labels)
            label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}
            gold = label_map.get(sample.get("label"))
        elif task == "copa":
            prompt = (
                "Premise: {premise}\n"
                "Question: What is the {question}?\n"
                "1) {choice1}\n"
                "2) {choice2}\n"
                "Answer with 1 or 2:"
            ).format(**sample)
            pred = bot.generate(prompt, generate_kwargs)
            pred_label = _extract_choice(pred or "")
            gold = "1" if sample.get("label") == 0 else "2"
        elif task == "wic":
            prompt = (
                "Sentence 1: {sentence1}\n"
                "Sentence 2: {sentence2}\n"
                "Does the word \"{word}\" have the same meaning in both sentences?\n"
                "Answer yes or no:"
            ).format(**sample)
            pred = bot.generate(prompt, generate_kwargs)
            pred_label = _extract_yes_no(pred or "")
            gold = "yes" if sample.get("label") == 1 else "no"
        else:
            return None, None

        if pred_label == gold:
            correct += 1
        total += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__superglue.json")

    data_root = "data"
    if hasattr(init_cfg, "data") and hasattr(init_cfg.data, "root"):
        data_root = init_cfg.data.root
    os.makedirs(data_root, exist_ok=True)

    tasks = _get_tasks(init_cfg)
    max_samples = _get_max_samples(init_cfg)
    generate_kwargs = dict(max_new_tokens=8,
                           do_sample=False,
                           temperature=1.0,
                           top_p=1.0)

    results = {}
    total_correct = 0
    total_seen = 0

    for task in tasks:
        try:
            dataset = load_dataset(
                "super_glue",
                task,
                split="validation",
                cache_dir=data_root,
            )
        except Exception as exc:
            print(f"[superglue] failed to load {task}: {exc}")
            continue
        acc, seen = _evaluate_task(task, dataset, bot, max_samples, generate_kwargs)
        results[task] = acc
        total_correct += acc * seen
        total_seen += seen

    weighted = total_correct / total_seen if total_seen > 0 else 0.0
    payload = {
        "tasks": tasks,
        "categories": results,
        "weighted_accuracy": weighted,
        "total_examples": total_seen,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"SuperGLUE results written to {out_path}")


if __name__ == "__main__":
    main()
