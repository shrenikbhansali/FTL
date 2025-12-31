#!/usr/bin/env python3
"""Run minimal sanity checks for FS-LLM training configs."""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_client_cfg
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.llm.dataset.tulu3_federated import load_tulu3_federated_data
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader import get_tokenizer, LLMDataCollator
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.eval.eval_for_gsm8k.eval import (
    build_prompt,
    clean_answer,
    extract_answer_from_output,
)


def _load_cfg(cfg_path: str) -> "Config":
    cfg = global_cfg.clone()
    cfg.merge_from_file(cfg_path)
    cfg_opt, _client_cfg = parse_client_cfg([])
    cfg.merge_from_list(cfg_opt)
    return cfg


def _load_model(cfg: "Config", load_ckpt: bool, precision: str, device: str):
    model = get_llm(cfg)
    ckpt_loaded = False
    ckpt_error = None
    if load_ckpt and cfg.federate.save_to:
        if os.path.exists(cfg.federate.save_to):
            try:
                ckpt = torch.load(cfg.federate.save_to, map_location="cpu")
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                model.load_state_dict(state, strict=False)
                ckpt_loaded = True
            except Exception as exc:
                ckpt_error = str(exc)
        else:
            ckpt_error = f"checkpoint not found: {cfg.federate.save_to}"
    model = model.to(device)
    if precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model = model.to(dtype=torch.float16)
    elif precision == "fp32":
        model = model.to(dtype=torch.float32)
    model.eval()
    return model, ckpt_loaded, ckpt_error


def _move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, tensor in batch.items():
        if torch.is_tensor(tensor):
            moved[key] = tensor.to(device)
        else:
            moved[key] = tensor
    return moved


def _check_batch(batch: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    labels = batch.get("labels")
    if labels is None:
        return False, "missing_labels"
    if torch.is_tensor(labels) and (labels == -100).all().item():
        return True, "all_labels_ignored"
    return False, ""


def _maybe_autocast(precision: str, device: str):
    enabled = device.startswith("cuda")
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False)


def _load_gsm8k_samples(data_root: str, max_samples: int = 2):
    gsm8k_path = os.path.join(data_root, "gsm8k_test.jsonl")
    if not os.path.exists(gsm8k_path):
        from federatedscope.core.data.utils import download_url
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            data_root,
        )
        os.rename(os.path.join(data_root, "test.jsonl"), gsm8k_path)
    samples = load_jsonl(gsm8k_path, instruction="question", output="answer")
    return samples[:max_samples]


def run_sanity(
    cfg_path: str,
    precision: str,
    load_ckpt: bool,
    max_batches: int,
    max_clients: int,
    do_train_step: bool,
    do_gsm8k_gen: bool,
) -> Dict:
    os.environ.setdefault("WANDB_DISABLED", "true")
    cfg = _load_cfg(cfg_path)
    cfg.defrost()
    if hasattr(cfg, "wandb"):
        cfg.wandb.use = False
    cfg.train.is_enable_half = False

    update_logger(cfg, clear_before_add=True)
    setup_seed(cfg.seed)

    data_dict, cfg = load_tulu3_federated_data(cfg, client_cfgs=None)
    device = f"cuda:{cfg.device}" if cfg.use_gpu else "cpu"
    model, ckpt_loaded, ckpt_error = _load_model(
        cfg, load_ckpt=load_ckpt, precision=precision, device=device
    )

    report = {
        "cfg_path": cfg_path,
        "precision": precision,
        "load_ckpt": load_ckpt,
        "device": device,
        "ckpt_loaded": ckpt_loaded,
        "ckpt_error": ckpt_error,
        "clients": [],
        "errors": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_step": {
            "attempted": False,
            "success": False,
            "skipped": False,
            "nan_loss": False,
            "nan_grad": False,
            "error": None,
        },
        "gsm8k_gen": {
            "attempted": False,
            "success": False,
            "error": None,
            "samples": [],
        },
    }

    client_ids = [cid for cid in data_dict.keys() if cid != 0]
    if max_clients > 0:
        client_ids = client_ids[:max_clients]

    model_name, model_hub = cfg.model.type.split("@")
    tokenizer, _ = get_tokenizer(model_name, cfg.data.root, cfg.llm.tok_len, model_hub)
    collator = LLMDataCollator(tokenizer=tokenizer)

    first_batch = None
    for client_id in client_ids:
        client = data_dict[client_id]
        train_dataset = getattr(client, "train_data", None)
        if train_dataset is None:
            report["errors"].append(
                {"client_id": client_id, "error": "missing_train_dataset"}
            )
            continue
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            drop_last=cfg.dataloader.drop_last,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=collator,
        )
        client_report = {
            "client_id": client_id,
            "num_samples": len(train_dataset),
            "checked_batches": 0,
            "nan_loss_batches": 0,
            "skipped_batches": 0,
            "errors": [],
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                client_report["checked_batches"] += 1
                try:
                    batch = _move_batch(batch, device)
                    skip, reason = _check_batch(batch)
                    if skip:
                        client_report["skipped_batches"] += 1
                        continue
                    with _maybe_autocast(precision, device):
                        outputs = model(**batch)
                    loss = outputs.loss
                    if loss is None or torch.isnan(loss).item() or torch.isinf(loss).item():
                        client_report["nan_loss_batches"] += 1
                except Exception as exc:
                    client_report["errors"].append(str(exc))
                if first_batch is None and isinstance(batch, dict):
                    first_batch = batch
        report["clients"].append(client_report)

    if do_train_step and first_batch is not None:
        report["train_step"]["attempted"] = True
        try:
            model.train()
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-6,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
            skip, _reason = _check_batch(first_batch)
            if skip:
                report["train_step"]["skipped"] = True
            else:
                optimizer.zero_grad(set_to_none=True)
                with _maybe_autocast(precision, device):
                    outputs = model(**first_batch)
                    loss = outputs.loss
                if loss is None or torch.isnan(loss).item() or torch.isinf(loss).item():
                    report["train_step"]["nan_loss"] = True
                if precision == "fp16":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                nan_grad = False
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_grad = True
                        break
                report["train_step"]["nan_grad"] = nan_grad
                if precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                report["train_step"]["success"] = not (
                    report["train_step"]["nan_loss"] or report["train_step"]["nan_grad"]
                )
        except Exception as exc:
            report["train_step"]["error"] = str(exc)
        finally:
            model.eval()

    if do_gsm8k_gen:
        report["gsm8k_gen"]["attempted"] = True
        try:
            samples = _load_gsm8k_samples(cfg.data.root, max_samples=2)
            for sample in samples:
                prompt = build_prompt(sample["instruction"], n_shot=8, cot_flag=True)
                with torch.no_grad():
                    inputs = tokenizer(
                        prompt,
                        padding=False,
                        add_special_tokens=True,
                        return_tensors="pt",
                    ).to(device)
                    completion_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        top_p=0.95,
                        temperature=0.8,
                        do_sample=True,
                    )
                input_len = inputs.input_ids.shape[1]
                completion_text = tokenizer.decode(
                    completion_ids[0][input_len:],
                    skip_special_tokens=True,
                    ignore_tokenization_space=True,
                )
                cleaned = clean_answer(completion_text)
                report["gsm8k_gen"]["samples"].append(
                    {
                        "question": sample["instruction"],
                        "answer": extract_answer_from_output(sample["output"]),
                        "completion": completion_text,
                        "cleaned_answer": cleaned,
                    }
                )
            report["gsm8k_gen"]["success"] = True
        except Exception as exc:
            report["gsm8k_gen"]["error"] = str(exc)

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to training YAML")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--load-ckpt", action="store_true", help="Load cfg.federate.save_to if present")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-clients", type=int, default=2)
    parser.add_argument("--do-train-step", action="store_true")
    parser.add_argument("--do-gsm8k-gen", action="store_true")
    args = parser.parse_args()

    report = run_sanity(
        cfg_path=args.cfg,
        precision=args.precision,
        load_ckpt=args.load_ckpt,
        max_batches=args.max_batches,
        max_clients=args.max_clients,
        do_train_step=args.do_train_step,
        do_gsm8k_gen=args.do_gsm8k_gen,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote sanity report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
