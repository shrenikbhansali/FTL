#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    return by_id


def fmt_num(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_section(title, metrics):
    lines = [f"## {title}", ""]

    server = metrics.get(0)
    sys_avg = metrics.get("sys_avg")
    sys_std = metrics.get("sys_std")

    lines.extend([
        "| scope | walltime (min) | total_flops | upload_bytes | download_bytes |",
        "| --- | --- | --- | --- | --- |",
    ])
    if server:
        lines.append(
            "| server | {w} | {f} | {u} | {d} |".format(
                w=fmt_num(server.get("fl_end_time_minutes")),
                f=fmt_num(server.get("total_flops")),
                u=fmt_num(server.get("total_upload_bytes")),
                d=fmt_num(server.get("total_download_bytes")),
            )
        )
    if sys_avg:
        lines.append(
            "| sys_avg | {w} | {f} | {u} | {d} |".format(
                w=fmt_num(sys_avg.get("sys_avg/fl_end_time_minutes")),
                f=fmt_num(sys_avg.get("sys_avg/total_flops")),
                u=fmt_num(sys_avg.get("sys_avg/total_upload_bytes")),
                d=fmt_num(sys_avg.get("sys_avg/total_download_bytes")),
            )
        )
    if sys_std:
        lines.append(
            "| sys_std | {w} | {f} | {u} | {d} |".format(
                w=fmt_num(sys_std.get("sys_std/fl_end_time_minutes")),
                f=fmt_num(sys_std.get("sys_std/total_flops")),
                u=fmt_num(sys_std.get("sys_std/total_upload_bytes")),
                d=fmt_num(sys_std.get("sys_std/total_download_bytes")),
            )
        )
    lines.append("")

    # Client details
    client_rows = [row for key, row in metrics.items() if isinstance(key, int) and key != 0]
    if client_rows:
        lines.extend([
            "### Per-client metrics",
            "",
            "| client_id | walltime (min) | total_flops | upload_bytes | download_bytes |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in sorted(client_rows, key=lambda r: r["id"]):
            lines.append(
                "| {cid} | {w} | {f} | {u} | {d} |".format(
                    cid=row["id"],
                    w=fmt_num(row.get("fl_end_time_minutes")),
                    f=fmt_num(row.get("total_flops")),
                    u=fmt_num(row.get("total_upload_bytes")),
                    d=fmt_num(row.get("total_download_bytes")),
                )
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize overhead system_metrics.log files.")
    parser.add_argument("--run-tag", required=True, help="Run tag under FTL/overhead (e.g., 20260126_175754)")
    parser.add_argument("--out", default="FTL/doc/overhead.md", help="Output markdown path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "overhead" / args.run_tag
    fedavg_log = run_dir / "fedavg" / "overhead_fedavg" / "system_metrics.log"
    bank_log = run_dir / "subspacebank_serveronly" / "overhead_subspacebank_serveronly" / "system_metrics.log"

    if not fedavg_log.exists() or not bank_log.exists():
        missing = [p for p in [fedavg_log, bank_log] if not p.exists()]
        raise SystemExit(f"Missing system_metrics.log: {missing}")

    fedavg = load_metrics(fedavg_log)
    bank = load_metrics(bank_log)

    md_lines = [
        "# Overhead results (H200)",
        "",
        f"Run tag: `{args.run_tag}`",
        "",
        "Configs:",
        "- FedAvg: `FTL/yamls/individual_federated_fedavg.yaml`",
        "- SubspaceBank (server-only): `FTL/yamls/individual_federated_bank_perclient.yaml` with overrides `train.unlearn.project_grads=False`, `aggregator.unlearn.send_Q_to_clients=False`",
        "",
        "Notes:",
        "- `total_flops` is only non-zero if `eval.count_flops=True` was set during the run.",
        "- `sys_avg`/`sys_std` are aggregated over all workers (server + clients).",
        "",
        build_section("FedAvg", fedavg),
        build_section("SubspaceBank (server-only)", bank),
        "Raw logs:",
        f"- FedAvg: `{fedavg_log}`",
        f"- SubspaceBank (server-only): `{bank_log}`",
        "",
    ]

    out_path = Path(args.out)
    out_path.write_text("\n".join(md_lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
