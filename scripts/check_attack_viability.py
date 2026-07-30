from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare attacked Baseline runs with clean controls. This script "
            "reports attack viability and never fabricates a pass threshold."
        )
    )
    parser.add_argument("--main", default="experiment_results_main_backdoor")
    parser.add_argument(
        "--controls",
        default="experiment_results_attack_viability_controls",
    )
    parser.add_argument(
        "--out",
        default="experiment_results_main_backdoor/attack_viability.csv",
    )
    args = parser.parse_args()

    attacked_files = sorted(Path(args.main).glob(
        "**/baseline/seed_*/fedagg_run_summary_*.csv"
    ))
    controls_root = Path(args.controls)
    control_files = sorted(controls_root.glob(
        "clean/**/seed_*/fedagg_run_summary_*.csv"
    ))
    if not control_files:
        # Backward compatibility with the original clean-only directory shape.
        control_files = sorted(controls_root.glob(
            "*/seed_*/fedagg_run_summary_*.csv"
        ))
    triggered_control_files = sorted(controls_root.glob(
        "triggered-no-poison/**/seed_*/fedagg_run_summary_*.csv"
    ))
    if not attacked_files or not control_files:
        raise FileNotFoundError(
            "Both attacked Baseline summaries and no-attack controls are required."
        )
    attacked = pd.concat([pd.read_csv(path) for path in attacked_files], ignore_index=True)
    control = pd.concat([pd.read_csv(path) for path in control_files], ignore_index=True)
    control = control[["dataset", "seed", "final_accuracy"]].rename(
        columns={"final_accuracy": "clean_control_accuracy"}
    )
    merged = attacked.merge(control, on=["dataset", "seed"], how="left", validate="many_to_one")
    if triggered_control_files:
        triggered_control = pd.concat(
            [pd.read_csv(path) for path in triggered_control_files],
            ignore_index=True,
        )
        triggered_control = triggered_control[[
            "dataset",
            "attack_type",
            "seed",
            "final_basr_global",
            "total_poisoned_samples",
        ]].rename(columns={
            "final_basr_global": "triggered_no_poison_basr",
            "total_poisoned_samples": "triggered_no_poison_poisoned_samples",
        })
        merged = merged.merge(
            triggered_control,
            on=["dataset", "attack_type", "seed"],
            how="left",
            validate="one_to_one",
        )
    else:
        merged["triggered_no_poison_basr"] = float("nan")
        merged["triggered_no_poison_poisoned_samples"] = float("nan")
    merged["clean_accuracy_drop"] = (
        merged["clean_control_accuracy"] - merged["final_accuracy"]
    )
    # Natural target prediction on clean controls is not BASR because no trigger
    # is applied for attack_type=none.  The attacked Baseline BASR therefore
    # remains the primary attack-establishment measurement.
    output = merged[[
        "dataset",
        "attack_type",
        "seed",
        "final_basr_global",
        "mean_attack_window_basr",
        "final_accuracy",
        "clean_control_accuracy",
        "clean_accuracy_drop",
        "total_poisoned_samples",
        "triggered_no_poison_basr",
        "triggered_no_poison_poisoned_samples",
    ]].copy()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))
    print(
        "Interpretation: a NIABD defense claim is valid only for configurations "
        "where the corresponding attacked Baseline establishes a substantial "
        "BASR. No numeric success threshold is hard-coded here."
    )
    print(
        "If the attacked Baseline does not establish a persistent backdoor: "
        "attack did not successfully propagate through the FD knowledge interface"
    )
    print(
        "If malicious and benign teachers remain similar on clean proxy logits: "
        "NIABD has insufficient observable evidence on clean proxy logits."
    )
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
