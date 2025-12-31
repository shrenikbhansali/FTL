#!/usr/bin/env python3
"""Quickly test one GSM8K-style prompt to isolate eval-time failures."""

import argparse
import sys

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot


PROMPT = (
    "Shawn has five toys. For Christmas, he got two toys each from his mom "
    "and dad. How many toys does he have now?"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to eval YAML")
    args, _ = parser.parse_known_args()

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(args.cfg)
    cfg_opt, _client_cfg = parse_client_cfg([])
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    fschatbot = FSChatBot(init_cfg)
    try:
        output = fschatbot.generate(
            PROMPT,
            generate_kwargs=dict(
                max_new_tokens=128,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
            ),
        )
        print("Generation succeeded.")
        print(output)
    except Exception as exc:
        print("Generation failed:", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
