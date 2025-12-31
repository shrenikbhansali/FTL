import json
import os
import sys

import transformers
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

IFEVAL_URL = (
    "https://huggingface.co/datasets/google/IFEval/"
    "resolve/main/ifeval_input_data.jsonl"
)
IFEVAL_FILE = "ifeval_input_data.jsonl"


def _ensure_open_instruct_on_path():
    env_root = os.environ.get("OPEN_INSTRUCT_ROOT")
    if env_root:
        open_instruct_root = env_root
    else:
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../")
        )
        open_instruct_root = os.path.join(root_dir, "materials", "open-instruct")
    if open_instruct_root not in sys.path:
        sys.path.insert(0, open_instruct_root)


def _load_ifeval(path, max_samples=None):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    _ensure_open_instruct_on_path()
    try:
        from open_instruct.IFEvalG import instructions_registry
    except Exception as exc:
        raise RuntimeError(
            "Failed to import open_instruct IFEval utilities. "
            "Ensure materials/open-instruct is available and its "
            "dependencies (e.g., langdetect, absl-py) are installed."
        ) from exc

    fschatbot = FSChatBot(init_cfg)

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)
    data_path = os.path.join(data_root, IFEVAL_FILE)
    if not os.path.exists(data_path):
        download_url(IFEVAL_URL, data_root)

    max_samples = None
    if hasattr(init_cfg, "eval") and hasattr(init_cfg.eval, "max_samples"):
        max_samples = init_cfg.eval.max_samples
    samples = _load_ifeval(data_path, max_samples=max_samples)

    max_new_tokens = 512
    if hasattr(init_cfg, "eval") and hasattr(init_cfg.eval, "max_new_tokens"):
        max_new_tokens = init_cfg.eval.max_new_tokens
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
    )

    scores = []
    strict_scores = []
    missing_instructions = set()

    for sample in tqdm(samples):
        prompt = sample.get("prompt", "")
        instruction_ids = sample.get("instruction_id_list", [])
        kwargs_list = sample.get("kwargs", [])
        if not instruction_ids:
            continue

        response = fschatbot.generate(prompt, generate_kwargs)
        per_instruction = []

        for instruction_id, raw_args in zip(instruction_ids, kwargs_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT.get(
                instruction_id
            )
            if instruction_cls is None:
                missing_instructions.add(instruction_id)
                per_instruction.append(0.0)
                continue

            args = raw_args or {}
            args = {k: v for k, v in args.items() if v is not None}
            try:
                instruction = instruction_cls(instruction_id)
                instruction.build_description(**args)
                per_instruction.append(
                    1.0 if instruction.check_following(response) else 0.0
                )
            except Exception:
                per_instruction.append(0.0)

        if not per_instruction:
            continue

        loose_score = sum(per_instruction) / len(per_instruction)
        strict_score = 1.0 if all(score == 1.0 for score in per_instruction) else 0.0
        scores.append(loose_score)
        strict_scores.append(strict_score)

    if not scores:
        print("No IFEval samples were processed.")
        return

    loose_acc = float(sum(scores) / len(scores))
    strict_acc = float(sum(strict_scores) / len(strict_scores))
    results = {
        "categories": {"ifeval": loose_acc},
        "weighted_accuracy": loose_acc,
        "strict_accuracy": strict_acc,
        "missing_instructions": sorted(missing_instructions),
    }

    eval_dir = "eval_result"
    if hasattr(init_cfg, "outdir") and init_cfg.outdir:
        eval_dir = os.path.join(init_cfg.outdir, "eval_result")
    os.makedirs(eval_dir, exist_ok=True)
    save_name = init_cfg.federate.save_to.replace("/", "_")
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__ifeval.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"IFEval results written to {out_path}")


if __name__ == "__main__":
    main()
