import argparse
import json
import os
from pathlib import Path

import collect_tulu_eval_results as collector


def _find_latest_run_dir(run_root: Path):
    if not run_root.exists():
        return None
    runs = [path for path in run_root.glob("run-*") if path.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1]


def _parse_wandb_run_id(run_dir: Path):
    if run_dir is None:
        return None
    parts = run_dir.name.split("-")
    if not parts:
        return None
    return parts[-1]


def _load_wandb_meta(run_dir: Path):
    if run_dir is None:
        return {}
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_backfill_for_exp(pipe_dir: Path, exp: str, project: str, entity: str):
    results_root = str(pipe_dir)
    aggregate = collector._collect_results(results_root, exp, "")
    out_dir = pipe_dir / exp / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregate.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(f"[backfill] wrote {out_path}")

    metrics = collector._flatten_wandb_metrics(aggregate)
    if exp == "bank_perclient":
        fedavg_aggregate = collector._collect_fedavg_results(
            results_root, ""
        )
        bank_primary = collector._extract_primary_metrics(aggregate)
        fedavg_primary = collector._extract_primary_metrics(fedavg_aggregate)
        collector._add_delta_metrics(metrics, bank_primary, fedavg_primary)

    run_root = pipe_dir / exp / "train" / "wandb" / "wandb"
    run_dir = _find_latest_run_dir(run_root)
    run_id = _parse_wandb_run_id(run_dir)
    if not run_id:
        print(f"[backfill] skip {pipe_dir.name}/{exp}: no wandb run id found")
        return

    meta = _load_wandb_meta(run_dir)
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run_id
    if project:
        env["WANDB_PROJECT"] = project
    elif meta.get("project"):
        env["WANDB_PROJECT"] = meta["project"]
    if entity:
        env["WANDB_ENTITY"] = entity
    elif meta.get("entity"):
        env["WANDB_ENTITY"] = meta["entity"]

    os.environ.update(env)
    collector._log_to_wandb(exp, metrics)
    print(f"[backfill] logged {pipe_dir.name}/{exp} to wandb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="tulupipe_results")
    parser.add_argument("--project", default="")
    parser.add_argument("--entity", default="")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    if not root.exists():
        raise SystemExit(f"results root not found: {root}")

    pipe_dirs = [path for path in root.iterdir() if path.is_dir()]
    pipe_dirs.sort()
    for pipe_dir in pipe_dirs:
        for exp in ("fedavg", "bank_perclient"):
            _run_backfill_for_exp(pipe_dir, exp, args.project, args.entity)


if __name__ == "__main__":
    main()
