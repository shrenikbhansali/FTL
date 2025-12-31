#!/usr/bin/env python3
"""Aggregate sanity check JSON reports into a single summary."""

import argparse
import json
import os
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    clients = report.get("clients", [])
    total_batches = sum(c.get("checked_batches", 0) for c in clients)
    total_skipped = sum(c.get("skipped_batches", 0) for c in clients)
    total_nan = sum(c.get("nan_loss_batches", 0) for c in clients)
    client_errors = [
        {"client_id": c.get("client_id"), "errors": c.get("errors", [])}
        for c in clients
        if c.get("errors")
    ]
    return {
        "cfg_path": report.get("cfg_path"),
        "precision": report.get("precision"),
        "load_ckpt": report.get("load_ckpt"),
        "ckpt_loaded": report.get("ckpt_loaded"),
        "ckpt_error": report.get("ckpt_error"),
        "device": report.get("device"),
        "total_clients": len(clients),
        "total_batches": total_batches,
        "total_skipped_batches": total_skipped,
        "total_nan_loss_batches": total_nan,
        "client_errors": client_errors,
        "train_step": report.get("train_step"),
        "gsm8k_gen": report.get("gsm8k_gen"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of sanity check JSON files.")
    parser.add_argument("--out", required=True, help="Output JSON summary file.")
    args = parser.parse_args()

    raw_reports: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        report = _load_json(path)
        raw_reports.append(report)
        summaries.append(_summarize(report))

    combined = {
        "reports": summaries,
        "total_reports": len(summaries),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
