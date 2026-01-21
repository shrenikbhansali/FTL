import json
import math
import os
import subprocess
import sys
import tempfile
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
    return textwrap.dedent(text).strip()


def _run_program(code: str, input_data: str, timeout: float) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
    except Exception:
        output = "__ERROR__"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return output


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
    return 5.0


def _load_io(sample):
    raw = sample.get("input_output")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    inputs = raw.get("inputs") or raw.get("input") or []
    outputs = raw.get("outputs") or raw.get("output") or []
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]
    if len(inputs) != len(outputs) or not inputs:
        return None
    return list(zip(inputs, outputs))


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
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__apps.json")

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

    try:
        dataset = load_dataset("codeparrot/apps", split=split, cache_dir=data_root)
    except Exception as exc:
        payload = {
            "error": f"Failed to load APPS dataset: {exc}",
            "pass@1": None,
            "pass@5": None,
            "pass@10": None,
            "total_examples": 0,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(payload["error"])
        print(f"APPS results written to {out_path}")
        return

    pass_at_1 = []
    pass_at_5 = []
    pass_at_10 = []
    total = 0

    for sample in tqdm(dataset, desc="apps"):
        if max_samples is not None and total >= max_samples:
            break
        prompt = sample.get("question")
        if not prompt:
            continue
        io_pairs = _load_io(sample)
        if not io_pairs:
            continue
        starter = sample.get("starter_code", "")
        completions = bot.generate(prompt, generate_kwargs)
        if isinstance(completions, str):
            completions = [completions]
        results = []
        for completion in completions:
            code = _clean_code(completion)
            if starter:
                code = starter + "\n" + code
            if not code.strip():
                results.append(False)
                continue
            passed_all = True
            for inp, expected in io_pairs:
                output = _run_program(code, inp, timeout)
                if output.strip() != str(expected).strip():
                    passed_all = False
                    break
            results.append(passed_all)
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
    print(f"APPS results written to {out_path}")


if __name__ == "__main__":
    main()
