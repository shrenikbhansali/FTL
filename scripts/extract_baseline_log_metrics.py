#!/usr/bin/env python3
"""Extract baseline log metrics into CSV + summary markdown."""

import argparse
import ast
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RESULTS_RE = re.compile(r"INFO: (\{.*\})")
CLIENT_RE = re.compile(r"Client #(?P<client>\d+)")
BANK_KEY_RE = re.compile(
    r"\[UNLEARN\]\[bank\] key=(?P<key>\S+) "
    r"global_frac=(?P<glob>[0-9.]+) "
    r"private_frac=(?P<priv>[0-9.]+) "
    r"resid_frac=(?P<resid>[0-9.]+) "
    r"alpha=(?P<alpha>[0-9.]+) "
    r"beta_global=(?P<beta_global>[0-9.]+) "
    r"beta_resid=(?P<beta_resid>[0-9.]+)"
)
BANK_AVG_RE = re.compile(
    r"\[UNLEARN\]\[bank\]\[avg\] "
    r"global_frac=(?P<glob>[0-9.]+) "
    r"private_frac=(?P<priv>[0-9.]+) "
    r"resid_frac=(?P<resid>[0-9.]+) "
    r"alpha=(?P<alpha>[0-9.]+) "
    r"beta_global=(?P<beta_global>[0-9.]+) "
    r"beta_resid=(?P<beta_resid>[0-9.]+)"
)
TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def parse_train_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = RESULTS_RE.search(line)
        if not match:
            continue
        payload = match.group(1)
        try:
            data = ast.literal_eval(payload)
        except Exception:
            continue
        if "Results_raw" not in data:
            continue
        role = data.get("Role", "")
        client_match = CLIENT_RE.search(role)
        if not client_match:
            continue
        client_id = int(client_match.group("client"))
        round_idx = int(data.get("Round", -1))
        results = data["Results_raw"]
        rows.append(
            {
                "round": round_idx,
                "client_id": client_id,
                "train_loss": float(results.get("train_loss", 0.0)),
                "train_total": int(results.get("train_total", 0)),
                "train_avg_loss": float(results.get("train_avg_loss", 0.0)),
                "train_nan_batches": int(results.get("train_nan_batches", 0)),
                "train_skipped_batches": int(results.get("train_skipped_batches", 0)),
            }
        )
    return rows


def parse_bank_geometry(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    frac_rows: List[Dict[str, object]] = []
    avg_rows: List[Dict[str, object]] = []
    round_idx = 0
    for line in path.read_text(errors="ignore").splitlines():
        key_match = BANK_KEY_RE.search(line)
        if key_match:
            frac_rows.append(
                {
                    "round": round_idx,
                    "key": key_match.group("key"),
                    "global_frac": float(key_match.group("glob")),
                    "private_frac": float(key_match.group("priv")),
                    "resid_frac": float(key_match.group("resid")),
                    "alpha": float(key_match.group("alpha")),
                    "beta_global": float(key_match.group("beta_global")),
                    "beta_resid": float(key_match.group("beta_resid")),
                }
            )
            continue
        avg_match = BANK_AVG_RE.search(line)
        if avg_match:
            ts_match = TS_RE.search(line)
            timestamp = ts_match.group("ts") if ts_match else ""
            avg_rows.append(
                {
                    "round": round_idx,
                    "timestamp": timestamp,
                    "global_frac": float(avg_match.group("glob")),
                    "private_frac": float(avg_match.group("priv")),
                    "resid_frac": float(avg_match.group("resid")),
                    "alpha": float(avg_match.group("alpha")),
                    "beta_global": float(avg_match.group("beta_global")),
                    "beta_resid": float(avg_match.group("beta_resid")),
                }
            )
            round_idx += 1
    return frac_rows, avg_rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_train(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {}
    avg_losses = [float(r["train_avg_loss"]) for r in rows]
    return {
        "rounds": len({int(r["round"]) for r in rows}),
        "rows": len(rows),
        "mean_train_avg_loss": statistics.mean(avg_losses),
        "min_train_avg_loss": min(avg_losses),
        "max_train_avg_loss": max(avg_losses),
        "total_nan_batches": sum(int(r["train_nan_batches"]) for r in rows),
        "total_skipped_batches": sum(int(r["train_skipped_batches"]) for r in rows),
    }


def summarize_bank_avg(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {}
    glob = [float(r["global_frac"]) for r in rows]
    priv = [float(r["private_frac"]) for r in rows]
    resid = [float(r["resid_frac"]) for r in rows]
    return {
        "rounds": len(rows),
        "mean_global_frac": statistics.mean(glob),
        "min_global_frac": min(glob),
        "max_global_frac": max(glob),
        "mean_private_frac": statistics.mean(priv),
        "min_private_frac": min(priv),
        "max_private_frac": max(priv),
        "mean_resid_frac": statistics.mean(resid),
        "min_resid_frac": min(resid),
        "max_resid_frac": max(resid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="individual_local_20260115_100457_21987",
        help="Baseline run id to parse.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_id = args.run_id
    results_root = root / "individual_results" / run_id
    out_dir = root / "final" / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def exp_path(method: str) -> Path:
        return results_root / method / "train" / f"{method}_run_{run_id}" / "exp_print.log"

    train_paths = {
        "centralized": exp_path("centralized"),
        "fedavg": exp_path("fedavg"),
        "bank_perclient": exp_path("bank_perclient"),
    }

    train_rows: Dict[str, List[TrainRow]] = {}
    for method, path in train_paths.items():
        if not path.exists():
            continue
        train_rows[method] = parse_train_rows(path)
        write_csv(
            out_dir / f"{method}_train_losses.csv",
            (
                {
                    "round": row["round"],
                    "client_id": row["client_id"],
                    "train_loss": row["train_loss"],
                    "train_total": row["train_total"],
                    "train_avg_loss": row["train_avg_loss"],
                    "train_nan_batches": row["train_nan_batches"],
                    "train_skipped_batches": row["train_skipped_batches"],
                }
                for row in train_rows[method]
            ),
            [
                "round",
                "client_id",
                "train_loss",
                "train_total",
                "train_avg_loss",
                "train_nan_batches",
                "train_skipped_batches",
            ],
        )

    bank_log = exp_path("bank_perclient")
    bank_frac_rows: List[Dict[str, object]] = []
    bank_avg_rows: List[Dict[str, object]] = []
    if bank_log.exists():
        bank_frac_rows, bank_avg_rows = parse_bank_geometry(bank_log)
        write_csv(
            out_dir / "bank_geometry_per_key.csv",
            (
                {
                    "round": row["round"],
                    "key": row["key"],
                    "global_frac": row["global_frac"],
                    "private_frac": row["private_frac"],
                    "resid_frac": row["resid_frac"],
                    "alpha": row["alpha"],
                    "beta_global": row["beta_global"],
                    "beta_resid": row["beta_resid"],
                }
                for row in bank_frac_rows
            ),
            [
                "round",
                "key",
                "global_frac",
                "private_frac",
                "resid_frac",
                "alpha",
                "beta_global",
                "beta_resid",
            ],
        )
        write_csv(
            out_dir / "bank_geometry_round_avg.csv",
            (
                {
                    "round": row["round"],
                    "timestamp": row["timestamp"],
                    "global_frac": row["global_frac"],
                    "private_frac": row["private_frac"],
                    "resid_frac": row["resid_frac"],
                    "alpha": row["alpha"],
                    "beta_global": row["beta_global"],
                    "beta_resid": row["beta_resid"],
                }
                for row in bank_avg_rows
            ),
            [
                "round",
                "timestamp",
                "global_frac",
                "private_frac",
                "resid_frac",
                "alpha",
                "beta_global",
                "beta_resid",
            ],
        )

    summary_path = out_dir / "baseline_log_summary.md"
    with summary_path.open("w") as handle:
        handle.write(f"# Baseline log summary\n\n")
        handle.write(f"Run id: `{run_id}`\n\n")
        handle.write("## Inputs\n")
        for method, path in train_paths.items():
            handle.write(f"- {method}: `{path}`\n")
        handle.write("\n")

        if bank_avg_rows:
            bank_summary = summarize_bank_avg(bank_avg_rows)
            handle.write("## Bank geometry fractions (round averages)\n")
            handle.write(f"Rounds: {bank_summary.get('rounds', 0)}\n\n")
            handle.write(
                "- global_frac: mean={mean_global_frac:.4f}, min={min_global_frac:.4f}, max={max_global_frac:.4f}\n".format(
                    **bank_summary
                )
            )
            handle.write(
                "- private_frac: mean={mean_private_frac:.4f}, min={min_private_frac:.4f}, max={max_private_frac:.4f}\n".format(
                    **bank_summary
                )
            )
            handle.write(
                "- resid_frac: mean={mean_resid_frac:.4f}, min={min_resid_frac:.4f}, max={max_resid_frac:.4f}\n".format(
                    **bank_summary
                )
            )
            handle.write("\n")
            handle.write(
                "Per-key fractions: `bank_geometry_per_key.csv` (" +
                f"{len(bank_frac_rows)} rows)\n\n"
            )
            handle.write("Round averages: `bank_geometry_round_avg.csv`\n\n")

        handle.write("## Training loss summaries\n")
        for method, rows in train_rows.items():
            summary = summarize_train(rows)
            if not summary:
                continue
            handle.write(f"### {method}\n")
            handle.write(f"Rounds: {summary['rounds']}\n\n")
            handle.write(
                "- train_avg_loss: mean={mean_train_avg_loss:.4f}, min={min_train_avg_loss:.4f}, max={max_train_avg_loss:.4f}\n".format(
                    **summary
                )
            )
            handle.write(
                "- total_nan_batches: {total_nan_batches}, total_skipped_batches: {total_skipped_batches}\n".format(
                    **summary
                )
            )
            handle.write(
                f"- CSV: `{method}_train_losses.csv` ({summary['rows']} rows)\n\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
