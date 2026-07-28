from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="experiment_results_main_backdoor")
    parser.add_argument(
        "--out",
        default="experiment_results_main_backdoor/main_backdoor_mean_std.csv",
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
        "final_accuracy",
        "final_basr_global",
        "mean_attack_window_basr",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {sorted(missing)}")
    duplicated = dataframe.duplicated(
        ["dataset", "attack_type", "strategy", "seed"],
        keep=False,
    )
    if bool(duplicated.any()):
        duplicates = dataframe.loc[
            duplicated,
            ["dataset", "attack_type", "strategy", "seed"],
        ]
        raise ValueError(
            "Duplicate formal runs detected:\n" + duplicates.to_string(index=False)
        )
    grouped = dataframe.groupby(
        ["dataset", "attack_type", "strategy"],
        sort=True,
    )
    table = grouped.agg(
        seeds=("seed", "count"),
        clean_acc_mean=("final_accuracy", "mean"),
        clean_acc_std=("final_accuracy", "std"),
        basr_mean=("final_basr_global", "mean"),
        basr_std=("final_basr_global", "std"),
        attack_window_basr_mean=("mean_attack_window_basr", "mean"),
        attack_window_basr_std=("mean_attack_window_basr", "std"),
        poisoned_samples_mean=("total_poisoned_samples", "mean"),
    ).reset_index()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"[write] {output}")


if __name__ == "__main__":
    main()
