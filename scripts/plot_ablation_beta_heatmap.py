#!/usr/bin/env python3
"""Plot beta sweep heatmap for ablation results."""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


BETA_RE = re.compile(r"abl_beta_g(?P<beta_g>[0-9]+(?:p5)?)_r(?P<beta_r>[0-9]+(?:p5)?)$")

DATASETS = {
    "gsm8k": ("weighted_accuracy", ("categories", "gsm8k")),
    "hellaswag": ("weighted_accuracy", ("categories", "hellaswag")),
    "xsum": ("rougeL_f1",),
    "hotpotqa": ("f1",),
    "mbpp": ("pass@1",),
}


def parse_beta(token: str) -> float:
    return float(token.replace("p", "."))


def load_metric(path: Path, dataset: str) -> float:
    data = json.loads(path.read_text())
    for key in DATASETS[dataset]:
        if isinstance(key, tuple):
            if key[0] in data and isinstance(data[key[0]], dict):
                value = data[key[0]].get(key[1])
                if value is not None:
                    return float(value)
        else:
            if key in data:
                return float(data[key])
    raise KeyError(f"Missing metric for {dataset} in {path}")


def find_eval_json(root: Path, dataset: str) -> Path:
    matches = list(root.rglob(f"*__{dataset}.json"))
    if not matches:
        raise FileNotFoundError(f"No eval json for {dataset} under {root}")
    if len(matches) > 1:
        matches = sorted(matches)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=(
            "/home/heck2/sbhansali8/FTL/final/results/ablations/"
            "individual_ablations_20260122_191942_5628/global"
        ),
        help="Root directory containing beta ablation result folders.",
    )
    parser.add_argument(
        "--output",
        default="/home/heck2/sbhansali8/FTL/doc/figs/ablation_beta_heatmap.pdf",
        help="Output path for the heatmap (pdf/png).",
    )
    parser.add_argument(
        "--title",
        default="Beta Sweep",
        help="Figure title.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(root)

    results = {}
    beta_g_vals = set()
    beta_r_vals = set()

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = BETA_RE.match(entry.name)
        if not match:
            continue
        beta_g = parse_beta(match.group("beta_g"))
        beta_r = parse_beta(match.group("beta_r"))
        beta_g_vals.add(beta_g)
        beta_r_vals.add(beta_r)

        metrics = []
        for dataset in DATASETS:
            dataset_root = entry / dataset
            eval_json = find_eval_json(dataset_root, dataset)
            metrics.append(load_metric(eval_json, dataset))
        macro = sum(metrics) / len(metrics)
        results[(beta_g, beta_r)] = macro

    beta_g_sorted = sorted(beta_g_vals)
    beta_r_sorted = sorted(beta_r_vals)

    grid = []
    for beta_g in beta_g_sorted:
        row = []
        for beta_r in beta_r_sorted:
            row.append(results.get((beta_g, beta_r), math.nan))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(grid, origin="lower", cmap="viridis")

    ax.set_xticks(range(len(beta_r_sorted)))
    ax.set_yticks(range(len(beta_g_sorted)))
    ax.set_xticklabels([str(v) for v in beta_r_sorted])
    ax.set_yticklabels([str(v) for v in beta_g_sorted])
    ax.set_xlabel(r"$\beta_r$")
    ax.set_ylabel(r"$\beta_g$")
    ax.set_title(args.title)

    for i, beta_g in enumerate(beta_g_sorted):
        for j, beta_r in enumerate(beta_r_sorted):
            value = results.get((beta_g, beta_r))
            if value is None or math.isnan(value):
                continue
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    if out_path.suffix.lower() != ".png":
        fig.savefig(out_path.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
