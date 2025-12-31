import argparse
import json
import os
from pathlib import Path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_task_file(task_dir, task):
    if task == "gsm8k":
        pattern = "accuracies_*__gsm8k.json"
    elif task == "humaneval":
        pattern = "accuracies_*__humaneval.json"
    else:
        pattern = "accuracies_*.json"

    matches = sorted(task_dir.glob(pattern))
    if task == "mmlu":
        matches = [
            path for path in matches
            if "__gsm8k" not in path.name and "__humaneval" not in path.name
        ]
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def _find_eval_dir(base_dir):
    if not base_dir.exists():
        return None
    candidates = [path for path in base_dir.rglob("eval_result")
                  if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _collect_results(results_root, exp, eval_job_id):
    task_files = {
        "mmlu": None,
        "gsm8k": None,
        "humaneval": None,
    }

    base_dir = Path(results_root) / "global" / exp / "{task}"
    if eval_job_id:
        base_dir = base_dir / eval_job_id
    benchmarks = {}
    missing = []

    for task in task_files:
        task_dir = Path(str(base_dir).format(task=task))
        eval_dir = _find_eval_dir(task_dir)
        if eval_dir is None:
            missing.append(str(task_dir))
            continue
        result_path = _find_task_file(eval_dir, task)
        if result_path is None:
            missing.append(str(eval_dir))
            continue
        benchmarks[task] = _load_json(result_path)

    aggregate = {
        "exp": exp,
        "eval_job_id": eval_job_id,
        "benchmarks": benchmarks,
        "missing": missing,
    }
    return aggregate


def _extract_primary_metrics(aggregate):
    metrics = {}
    benchmarks = aggregate.get("benchmarks", {})

    mmlu = benchmarks.get("mmlu", {})
    weighted = mmlu.get("weighted_accuracy")
    if weighted is not None:
        metrics["mmlu_weighted_accuracy"] = float(weighted)

    gsm8k = benchmarks.get("gsm8k", {})
    weighted = gsm8k.get("weighted_accuracy")
    if weighted is not None:
        metrics["gsm8k_weighted_accuracy"] = float(weighted)

    humaneval = benchmarks.get("humaneval", {})
    for key in ("pass@1", "pass@5", "pass@10"):
        if key in humaneval:
            metrics["humaneval_pass"] = float(humaneval[key])
            break

    return metrics


def _collect_fedavg_results(results_root, eval_job_id):
    try:
        return _collect_results(results_root, "fedavg", eval_job_id)
    except Exception:
        return {"benchmarks": {}, "missing": ["fedavg metrics unavailable"]}


def _flatten_wandb_metrics(aggregate):
    metrics = {}
    benchmarks = aggregate.get("benchmarks", {})

    mmlu = benchmarks.get("mmlu", {})
    if mmlu:
        weighted = mmlu.get("weighted_accuracy")
        if weighted is not None:
            metrics["benchmarks/mmlu/weighted_accuracy"] = weighted
        for cat, val in mmlu.get("categories", {}).items():
            metrics[f"benchmarks/mmlu/categories/{cat}"] = val

    gsm8k = benchmarks.get("gsm8k", {})
    if gsm8k:
        weighted = gsm8k.get("weighted_accuracy")
        if weighted is not None:
            metrics["benchmarks/gsm8k/weighted_accuracy"] = weighted

    humaneval = benchmarks.get("humaneval", {})
    if humaneval:
        for key, val in humaneval.items():
            if key.startswith("pass@"):
                metrics[f"benchmarks/humaneval/{key}"] = val

    if aggregate.get("missing"):
        metrics["benchmarks/missing_count"] = len(aggregate["missing"])

    return metrics


def _add_delta_metrics(metrics, bank_metrics, fedavg_metrics):
    deltas = {}
    keys = set(bank_metrics) & set(fedavg_metrics)
    task_deltas = []
    for key in keys:
        delta = bank_metrics[key] - fedavg_metrics[key]
        deltas[f"benchmarks/delta/{key}"] = delta
        task_deltas.append(delta)
    if task_deltas:
        deltas["benchmarks/delta/mean"] = sum(task_deltas) / len(task_deltas)
        deltas["benchmarks/delta/bank_better_rate"] = sum(
            1 for val in task_deltas if val > 0
        ) / len(task_deltas)
    metrics.update(deltas)


def _prefix_metrics(metrics, prefix, strip_root="benchmarks/"):
    prefixed = {}
    for key, val in metrics.items():
        if key.startswith(strip_root):
            new_key = prefix + key[len(strip_root):]
        else:
            new_key = prefix + key
        prefixed[new_key] = val
    return prefixed


def _log_to_wandb(exp, metrics):
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[collector] wandb unavailable: {exc}")
        return

    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="allow",
    )
    if metrics:
        wandb.log(metrics, step=0)
        for key, val in metrics.items():
            try:
                wandb.summary[key] = val
            except Exception:
                pass
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--eval-job-id", required=True)
    parser.add_argument("--fedavg-eval-job-id", default="")
    args = parser.parse_args()

    aggregate = _collect_results(args.results_root, args.exp, args.eval_job_id)

    out_dir = Path(args.results_root) / args.exp / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    print(f"[collector] wrote {out_path}")

    metrics = _flatten_wandb_metrics(aggregate)
    if args.exp == "bank_perclient":
        fedavg_aggregate = _collect_fedavg_results(
            args.results_root, args.fedavg_eval_job_id
        )
        fedavg_metrics = _flatten_wandb_metrics(fedavg_aggregate)
        metrics.update(_prefix_metrics(fedavg_metrics,
                                       "benchmarks/fedavg/"))
        bank_primary = _extract_primary_metrics(aggregate)
        fedavg_primary = _extract_primary_metrics(fedavg_aggregate)
        _add_delta_metrics(metrics, bank_primary, fedavg_primary)
    _log_to_wandb(args.exp, metrics)


if __name__ == "__main__":
    main()
