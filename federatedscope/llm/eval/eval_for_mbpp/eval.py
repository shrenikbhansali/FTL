import json
import math
import multiprocessing as mp
import os
import re
import textwrap

import transformers
from datasets import load_dataset
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
transformers.logging.set_verbosity(40)


def _clean_code(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            lines = text.splitlines()
            if lines and re.match(r"^\s*python\s*$", lines[0], re.IGNORECASE):
                lines = lines[1:]
            text = "\n".join(lines)
    return textwrap.dedent(text).strip()


def _worker(code: str, tests: str, queue: mp.Queue):
    try:
        namespace = {}
        exec(code, namespace)
        exec(tests, namespace)
        queue.put(True)
    except Exception:
        queue.put(False)


def _run_tests(code: str, tests: str, timeout: float) -> bool:
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(code, tests, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    if not queue.empty():
        return bool(queue.get())
    return False


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _get_max_samples(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_samples"):
        return int(cfg.eval.max_samples)
    return None


def _get_max_new_tokens(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "max_new_tokens"):
        return int(cfg.eval.max_new_tokens)
    return 256


def _get_num_completions(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "num_completions"):
        return int(cfg.eval.num_completions)
    return 1


def _get_timeout(cfg):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "timeout"):
        return float(cfg.eval.timeout)
    return 3.0


def _get_eval_opt(cfg, name, default):
    if hasattr(cfg, "eval") and hasattr(cfg.eval, name):
        return getattr(cfg.eval, name)
    return default


def _extract_signature(code: str, tests) -> tuple:
    if code:
        match = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:",
                          code,
                          re.MULTILINE)
        if match:
            name = match.group(1)
            args = match.group(2).strip()
            signature = f"def {name}({args}):"
            return signature, name
    text = ""
    if isinstance(tests, (list, tuple)):
        text = "\n".join(tests)
    elif isinstance(tests, str):
        text = tests
    match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(",
                      text,
                      re.MULTILINE)
    if match:
        return None, match.group(1)
    return None, None


def _build_user_prompt(text: str,
                       signature: str,
                       func_name: str,
                       include_instructions: bool = True) -> str:
    parts = []
    if include_instructions:
        parts.append("Write a Python function that solves the following problem.")
        parts.append("Do not use input() or print(). Return the result.")
    if text:
        parts.append(text.strip())
    if signature:
        parts.append(f"Function signature:\n{signature}")
    elif func_name:
        parts.append(f"Function name: {func_name}")
    parts.append("Provide only the Python code.")
    return "\n\n".join(parts)


def _load_prompt_examples(cfg, data_root):
    num_shots = int(_get_eval_opt(cfg, "mbpp_num_shots", 0))
    if num_shots <= 0:
        return []
    split = "prompt"
    config_name = _get_eval_opt(cfg, "mbpp_config", None)
    dataset = load_dataset("mbpp",
                           config_name,
                           split=split,
                           cache_dir=data_root)
    examples = []
    for sample in dataset:
        text = sample.get("text")
        code = sample.get("code")
        tests = sample.get("test_list") or sample.get("tests")
        if not text or not code:
            continue
        signature, func_name = _extract_signature(code, tests)
        user_prompt = _build_user_prompt(text, signature, func_name, True)
        examples.append((user_prompt, str(code).rstrip()))
        if len(examples) >= num_shots:
            break
    return examples


def _build_prompt(sample, cfg, prompt_examples):
    text = sample.get("text")
    code = sample.get("code")
    tests = sample.get("test_list") or sample.get("tests")
    signature, func_name = _extract_signature(str(code) if code else "", tests)
    user_prompt = _build_user_prompt(text, signature, func_name, True)
    use_chat = bool(_get_eval_opt(cfg, "mbpp_use_chat_prompt", True))
    if use_chat:
        parts = []
        for shot_prompt, shot_code in prompt_examples:
            parts.append(f"User: {shot_prompt}\n\nAssistant: {shot_code}")
        parts.append(f"User: {user_prompt}\n\nAssistant:")
        return "\n\n".join(parts)
    parts = []
    for shot_prompt, shot_code in prompt_examples:
        parts.append("### Problem\n" + shot_prompt + "\n### Solution\n" + shot_code)
    parts.append("### Problem\n" + user_prompt + "\n### Solution\n")
    return "\n\n".join(parts)


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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__mbpp.json")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)

    max_samples = _get_max_samples(init_cfg)
    max_new_tokens = _get_max_new_tokens(init_cfg)
    num_completions = _get_num_completions(init_cfg)
    timeout = _get_timeout(init_cfg)
    generate_kwargs = dict(max_new_tokens=max_new_tokens,
                           temperature=0.2,
                           top_p=0.95,
                           do_sample=num_completions > 1,
                           num_return_sequences=num_completions)

    split = "test"
    if hasattr(init_cfg, "eval") and hasattr(init_cfg.eval, "split"):
        split_val = init_cfg.eval.split
        if isinstance(split_val, str):
            split = split_val
        elif isinstance(split_val, (list, tuple)) and split_val:
            split = split_val[0]

    config_name = _get_eval_opt(init_cfg, "mbpp_config", None)
    dataset = load_dataset("mbpp",
                           config_name,
                           split=split,
                           cache_dir=data_root)
    prompt_examples = _load_prompt_examples(init_cfg, data_root)

    pass_at_1 = []
    pass_at_5 = []
    pass_at_10 = []
    total = 0

    for sample in tqdm(dataset, desc="mbpp"):
        if max_samples is not None and total >= max_samples:
            break
        prompt = _build_prompt(sample, init_cfg, prompt_examples)
        tests = sample.get("test_list") or sample.get("tests")
        if not prompt or not tests:
            continue
        setup = sample.get("test_setup_code", "")
        test_code = setup + "\n" + "\n".join(tests)
        completions = bot.generate(prompt, generate_kwargs)
        if isinstance(completions, str):
            completions = [completions]
        results = []
        for completion in completions:
            code = _clean_code(completion)
            if not code:
                results.append(False)
                continue
            results.append(_run_tests(code, test_code, timeout))
        n = len(results)
        c = sum(1 for r in results if r)
        if n == 0:
            continue
        pass_at_1.append(_pass_at_k(n, c, 1))
        pass_at_5.append(_pass_at_k(n, c, min(5, n)))
        pass_at_10.append(_pass_at_k(n, c, min(10, n)))
        total += 1

    payload = {
        "pass@1": float(sum(pass_at_1) / len(pass_at_1)) if pass_at_1 else 0.0,
        "pass@5": float(sum(pass_at_5) / len(pass_at_5)) if pass_at_5 else 0.0,
        "pass@10": float(sum(pass_at_10) / len(pass_at_10)) if pass_at_10 else 0.0,
        "total_examples": total,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"MBPP results written to {out_path}")


if __name__ == "__main__":
    main()
