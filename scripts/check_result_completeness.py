from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path


REQUIRED_COLUMNS = {
    "dataset",
    "attack_type",
    "strategy",
    "seed",
    "runtime",
    "rounds",
    "final_accuracy",
    "final_loss",
    "niabd_algorithm_version",
    "result_schema_version",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed completeness check for versioned formal results."
    )
    parser.add_argument("--indir", default="experiment_results_main_backdoor")
    parser.add_argument(
        "--config",
        default="configs/main_backdoor_experiment.json",
    )
    args = parser.parse_args()

    root = Path(args.indir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected = {
        (str(dataset), str(attack), str(method), int(seed))
        for dataset, attack, method, seed in product(
            config["datasets"],
            config["attacks"],
            config["methods"],
            config["seeds"],
        )
    }
    files = sorted(root.glob("**/fedagg_run_summary_*.csv"))
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise SystemExit(
                    f"FAIL: {path} is missing columns {sorted(missing)}"
                )
            for row in reader:
                key = (
                    row["dataset"],
                    row["attack_type"],
                    row["strategy"],
                    int(row["seed"]),
                )
                if row["final_accuracy"] == "" or row["final_loss"] == "":
                    raise SystemExit(f"FAIL: incomplete numeric result row in {path}")
                rows.append((key, row, path))

    actual = [key for key, _, _ in rows]
    duplicates = sorted({key for key in actual if actual.count(key) > 1})
    if duplicates:
        raise SystemExit(f"FAIL: duplicate formal jobs: {duplicates}")
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    if missing or unexpected:
        raise SystemExit(
            f"FAIL: missing={missing} unexpected={unexpected}"
        )
    if len(actual) != len(expected):
        raise SystemExit(
            f"FAIL: expected {len(expected)} rows, found {len(actual)}"
        )

    print(
        f"PASS: {len(actual)} complete jobs; no duplicates or missing jobs; "
        "algorithm/schema columns present and results remain version-addressable."
    )


if __name__ == "__main__":
    main()
