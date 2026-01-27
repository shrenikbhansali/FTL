#!/usr/bin/env python3
"""Plot baseline geometry fractions over rounds."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_round_avgs(path: Path):
    rounds = []
    glob = []
    priv = []
    resid = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rounds.append(int(row["round"]))
            glob.append(float(row["global_frac"]))
            priv.append(float(row["private_frac"]))
            resid.append(float(row["resid_frac"]))
    return rounds, glob, priv, resid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/home/heck2/sbhansali8/FTL/final/logs/individual_local_20260115_100457_21987/bank_geometry_round_avg.csv",
        help="CSV with round-averaged geometry fractions.",
    )
    parser.add_argument(
        "--output",
        default="/home/heck2/sbhansali8/FTL/final/plots/main/geometry_fractions_baseline.pdf",
        help="Output plot path (pdf/png).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rounds, glob, priv, _resid = load_round_avgs(in_path)

    plt.figure(figsize=(6.2, 3.4))
    plt.plot(rounds, glob, label="global fraction", linewidth=1.8)
    plt.plot(rounds, priv, label="private fraction", linewidth=1.8)
    plt.xlabel("Round")
    plt.ylabel("Energy fraction")
    plt.ylim(0.0, 1.0)
    plt.xlim(min(rounds), max(rounds))
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend(frameon=False, ncol=2, fontsize=8, loc="upper center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


if __name__ == "__main__":
    main()
