import json
import os
import shlex
import subprocess

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger


def _write_payload(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    eval_dir = "eval_result"
    if hasattr(init_cfg, "outdir") and init_cfg.outdir:
        eval_dir = os.path.join(init_cfg.outdir, "eval_result")
    os.makedirs(eval_dir, exist_ok=True)
    save_name = init_cfg.federate.save_to.replace("/", "_")
    out_path = os.path.join(eval_dir, f"accuracies_{save_name}__toolbench.json")

    cmd = os.environ.get("TOOLBENCH_EVAL_CMD")
    result_json = os.environ.get("TOOLBENCH_EVAL_OUTPUT")
    if not cmd or not result_json:
        payload = {
            "error": "TOOLBENCH_EVAL_CMD/TOOLBENCH_EVAL_OUTPUT not set",
            "score": None,
        }
        _write_payload(out_path, payload)
        print(payload["error"])
        print(f"ToolBench results written to {out_path}")
        return

    try:
        completed = subprocess.run(
            shlex.split(cmd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        payload = {
            "error": f"Failed to run ToolBench eval command: {exc}",
            "score": None,
        }
        _write_payload(out_path, payload)
        print(payload["error"])
        print(f"ToolBench results written to {out_path}")
        return

    if completed.returncode != 0:
        payload = {
            "error": f"ToolBench eval failed: {completed.stderr.strip()}",
            "score": None,
        }
        _write_payload(out_path, payload)
        print(payload["error"])
        print(f"ToolBench results written to {out_path}")
        return

    if not os.path.exists(result_json):
        payload = {
            "error": f"ToolBench eval output not found: {result_json}",
            "score": None,
        }
        _write_payload(out_path, payload)
        print(payload["error"])
        print(f"ToolBench results written to {out_path}")
        return

    with open(result_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    _write_payload(out_path, payload)
    print(f"ToolBench results written to {out_path}")


if __name__ == "__main__":
    main()
