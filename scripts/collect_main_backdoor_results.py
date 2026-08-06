from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from compute_statistics import _ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="experiment_results_main_backdoor")
    parser.add_argument(
        "--out",
        default="experiment_results_main_backdoor/main_backdoor_mean_std.csv",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=0,
        help="Require this many complete run rows when non-zero (formal matrix: 240).",
    )
    args = parser.parse_args()

    root = Path(args.indir)
    files = sorted(root.glob("**/fedagg_run_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No run summary CSV files found under {root}.")
    frames = [pd.read_csv(path) for path in files]
    dataframe = pd.concat(frames, ignore_index=True)
    required = {
        "dataset",
        "attack_type",
        "strategy",
        "seed",
        "runtime",
        "niabd_algorithm_version",
        "result_schema_version",
        "final_accuracy",
        "final_basr_global",
        "mean_attack_window_basr",
        "peak_attack_window_basr",
        "attack_window_basr_auc",
        "post_attack_recovery_basr",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {sorted(missing)}")
    if int(args.expected_runs) > 0 and len(dataframe) != int(args.expected_runs):
        raise ValueError(
            f"Expected {args.expected_runs} complete runs, found {len(dataframe)}."
        )
    identity = [
        "dataset",
        "attack_type",
        "strategy",
        "seed",
        "runtime",
        "niabd_algorithm_version",
        "result_schema_version",
    ]
    duplicated = dataframe.duplicated(
        identity,
        keep=False,
    )
    if bool(duplicated.any()):
        duplicates = dataframe.loc[
            duplicated,
            identity,
        ]
        raise ValueError(
            "Duplicate formal runs detected:\n" + duplicates.to_string(index=False)
        )
    grouped = dataframe.groupby(
        [
            "dataset",
            "attack_type",
            "strategy",
            "runtime",
            "niabd_algorithm_version",
            "result_schema_version",
        ],
        sort=True,
    )
    def ci95(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        if len(values) <= 1:
            return float("nan")
        _, lower, upper = _ci(values, 0.95)
        return float((upper - lower) / 2.0)

    table = grouped.agg(
        runs=("seed", "count"),
        clean_acc_mean=("final_accuracy", "mean"),
        clean_acc_std=("final_accuracy", "std"),
        basr_mean=("final_basr_global", "mean"),
        basr_std=("final_basr_global", "std"),
        attack_window_basr_mean=("mean_attack_window_basr", "mean"),
        attack_window_basr_std=("mean_attack_window_basr", "std"),
        attack_window_peak_mean=("peak_attack_window_basr", "mean"),
        attack_window_auc_mean=("attack_window_basr_auc", "mean"),
        post_attack_recovery_mean=("post_attack_recovery_basr", "mean"),
        poisoned_samples_mean=("total_poisoned_samples", "mean"),
    ).reset_index()
    ci_rows = []
    group_keys = [
        "dataset",
        "attack_type",
        "strategy",
        "runtime",
        "niabd_algorithm_version",
        "result_schema_version",
    ]
    for key, frame in grouped:
        ci_rows.append({
            **dict(zip(group_keys, key)),
            "clean_acc_ci95": ci95(frame["final_accuracy"]),
            "basr_ci95": ci95(frame["final_basr_global"]),
            "attack_window_basr_ci95": ci95(frame["mean_attack_window_basr"]),
        })
    grouped_ci = pd.DataFrame(ci_rows)
    table = table.merge(
        grouped_ci,
        on=[
            "dataset",
            "attack_type",
            "strategy",
            "runtime",
            "niabd_algorithm_version",
            "result_schema_version",
        ],
        how="left",
        validate="one_to_one",
    )
    table["missing_runs"] = (
        5 - table["runs"]
        if int(args.expected_runs) in {0, 240}
        else float("nan")
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"[write] {output}")


if __name__ == "__main__":
    main()
