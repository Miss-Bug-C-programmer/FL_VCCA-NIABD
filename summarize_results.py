from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


GROUP_COLUMNS = [
    "dataset",
    "runtime",
    "strategy",
    "topology",
    "server_model",
    "client_model",
    "server_device",
    "client_device",
    "num_clients",
    "partition_scheme",
    "knowledge_interface",
    "aggregation_rule",
    "vcaa_enabled",
    "admission_method",
    "niabd_enabled",
    "defense_method",
]

METRIC_COLUMNS = [
    "final_accuracy",
    "best_accuracy",
    "final_loss",
    "wall_clock_time_s",
    "total_rollbacks",
    "total_numeric_failures",
    "mean_teacher_utilization",
    "total_teachers_admitted",
    "total_teachers_rejected",
    "total_client_upload_bytes",
    "total_server_broadcast_bytes",
    "total_teachers_purified",
    "mean_niabd_anomaly_fraction",
    "mean_niabd_suppression",
    "final_niabd_threshold_mean",
    "total_niabd_prototype_updates",
    "total_client_wire_bytes",
    "total_packets_consumed",
    "total_stale_packets",
    "max_version_lag",
    "stale_rejection_rate",
    "fresh_rejection_rate",
]


def summarize(indir: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(os.path.join(indir, "fedagg_run_summary_*.csv"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No fedagg_run_summary_*.csv files found in {indir!r}."
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = [
        column
        for column in [*GROUP_COLUMNS, *METRIC_COLUMNS]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Run summary is missing required columns: {missing}")

    grouped = frame.groupby(GROUP_COLUMNS, dropna=False)
    rows = []
    for keys, group in grouped:
        row = dict(zip(GROUP_COLUMNS, keys))
        row["runs"] = int(len(group))
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            observed = values.dropna()
            row[f"{metric}_mean"] = (
                float(observed.mean()) if not observed.empty else float("nan")
            )
            row[f"{metric}_std"] = (
                float(observed.std(ddof=1))
                if len(observed) > 1
                else (0.0 if len(observed) == 1 else float("nan"))
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(GROUP_COLUMNS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize server-client FedAgg seed runs."
    )
    parser.add_argument("--indir", required=True)
    parser.add_argument("--latex-ready", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.indir)
    output = os.path.join(args.indir, "summary_server_client.csv")
    summary.to_csv(output, index=False)
    print(f"[write] {output}")

    if args.latex_ready:
        latex_output = os.path.join(
            args.indir,
            "summary_server_client_latex_ready.csv",
        )
        summary.to_csv(latex_output, index=False, float_format="%.4f")
        print(f"[write] {latex_output}")


if __name__ == "__main__":
    main()
