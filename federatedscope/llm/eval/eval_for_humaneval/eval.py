import json
import os

import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = False
NUM_ANSWERS_PER_QUESTION = 5


def clean_answer(code):
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    code = code.replace("\u00a0", "")
    for stop_seq in ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\nassert"]:
        code = code.split(stop_seq)[0]
    return pad_spaces(code, 4)


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _evaluate_with_human_eval(samples_path):
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except Exception:
        return None, (
            "human_eval is not installed. Install it via "
            "`pip install -e human-eval` to compute pass@k."
        )

    results = evaluate_functional_correctness(
        samples_path, k=[1, 5, 10], timeout=3.0
    )
    return results, None


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    fschatbot = FSChatBot(init_cfg)
    eval_dir = "eval_result"
    if hasattr(init_cfg, "outdir") and init_cfg.outdir:
        eval_dir = os.path.join(init_cfg.outdir, "eval_result")
    os.makedirs(eval_dir, exist_ok=True)
    save_name = init_cfg.federate.save_to.replace("/", "_")
    out_file = os.path.join(eval_dir, f"{save_name}_humaneval_answer.jsonl")

    data_root = init_cfg.data.root if hasattr(init_cfg, "data") else "data"
    os.makedirs(data_root, exist_ok=True)
    data_path = os.path.join(data_root, "HumanEval.jsonl.gz")
    if not os.path.exists(data_path):
        download_url(
            "https://github.com/openai/human-eval/raw/"
            "463c980b59e818ace59f6f9803cd92c749ceae61/"
            "data/HumanEval.jsonl.gz",
            data_root,
        )

    list_data_dict = load_jsonl(
        data_path,
        instruction="prompt",
        input="entry_point",
        category="task_id",
        output="test",
        is_gzip=True,
    )
    if hasattr(init_cfg, "eval") and hasattr(init_cfg.eval, "max_samples"):
        list_data_dict = list_data_dict[:init_cfg.eval.max_samples]

    answers = []
    generation_config = GenerationConfig(
        temperature=0.1,
        top_k=40,
        top_p=0.75,
        do_sample=True,
        num_return_sequences=NUM_ANSWERS_PER_QUESTION,
    )
    max_new_tokens = 128
    if hasattr(init_cfg, "eval") and hasattr(init_cfg.eval, "max_new_tokens"):
        max_new_tokens = init_cfg.eval.max_new_tokens
    generate_kwargs = dict(
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )

    for sample in tqdm(list_data_dict):
        input_text = sample["instruction"]
        try:
            model_completions = fschatbot.generate(input_text, generate_kwargs)
        except Exception as error:
            print(error)
            model_completions = [
                "" for _ in range(NUM_ANSWERS_PER_QUESTION)
            ]
        if isinstance(model_completions, str):
            model_completions = [model_completions]

        for i, completion in enumerate(model_completions):
            completion = clean_answer(completion)
            answers.append(
                dict(task_id=sample["category"], completion=completion)
            )
            if DEBUG:
                print(
                    f"task_id: {sample['category']},\n"
                    f"completion {i + 1}:\n{completion}\n\n"
                )

    with open(out_file, "w", encoding="utf-8") as f:
        for answer in answers:
            json_str = json.dumps(answer)
            f.write(json_str + "\n")

    results, error = _evaluate_with_human_eval(out_file)
    if results is None:
        print(error)
        return

    out_path = os.path.join(
        eval_dir, f"accuracies_{save_name}__humaneval.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"HumanEval results written to {out_path}")


if __name__ == "__main__":
    main()
