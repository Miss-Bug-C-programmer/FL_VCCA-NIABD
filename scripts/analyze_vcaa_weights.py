"""Analyze exported VCAA teacher weights without modifying experiment data.

This is an offline reporting utility.  Its optional malicious/benign split is
explicitly an oracle diagnostic and is never imported by the runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "dataset",
    "attack_type",
    "strategy",
    "seed",
    "round",
]


def _percentiles(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {f"{prefix}_{name}": float("nan") for name in ("min", "p25", "median", "p75", "max")}
    return {
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_p25": float(numeric.quantile(0.25)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_p75": float(numeric.quantile(0.75)),
        f"{prefix}_max": float(numeric.max()),
    }


def analyze(frame: pd.DataFrame, *, include_oracle: bool = False) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"teacher admission CSV is missing identity columns: {missing}")
    if "aggregation_weight" not in frame.columns:
        raise ValueError("teacher admission CSV is missing aggregation_weight")

    working = frame.copy()
    working["aggregation_weight"] = pd.to_numeric(
        working["aggregation_weight"], errors="coerce"
    )
    if "normalized_aggregation_weight" in working.columns:
        working["normalized_aggregation_weight"] = pd.to_numeric(
            working["normalized_aggregation_weight"], errors="coerce"
        )
    else:
        working["normalized_aggregation_weight"] = np.nan
    if "hard_valid" in working.columns:
        working["hard_valid"] = working["hard_valid"].astype(str).str.lower().isin(
            {"true", "1", "1.0"}
        )
    else:
        working["hard_valid"] = False

    rows = []
    for key, group in working.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        valid = group[group["hard_valid"]]
        raw = valid["aggregation_weight"].replace([np.inf, -np.inf], np.nan).dropna()
        raw = raw[raw > 0.0]
        normalized = valid["normalized_aggregation_weight"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if normalized.empty and not raw.empty:
            normalized = raw / float(raw.sum())
        row = dict(zip(IDENTITY_COLUMNS, key))
        row["freshness_valid_teachers"] = int(len(valid))
        row.update(_percentiles(raw, "raw_weight"))
        row.update(_percentiles(normalized, "normalized_weight"))
        row.update(_percentiles(valid.get("content_score", pd.Series(dtype=float)), "content_score"))
        row.update(_percentiles(valid.get("content_reliability", pd.Series(dtype=float)), "content_reliability"))
        if normalized.empty:
            row.update({
                "effective_teacher_count": float("nan"),
                "weight_cv": float("nan"),
                "weight_total_variation_from_uniform": float("nan"),
                "content_reliability_saturation_fraction": float("nan"),
            })
        else:
            values = normalized.to_numpy(dtype=float)
            uniform = 1.0 / len(values)
            row.update({
                "effective_teacher_count": float(1.0 / np.square(values).sum()),
                "weight_cv": float(np.std(values) / np.mean(values)),
                "weight_total_variation_from_uniform": float(
                    0.5 * np.abs(values - uniform).sum()
                ),
                "content_reliability_saturation_fraction": float(
                    (pd.to_numeric(valid["content_reliability"], errors="coerce") >= 0.999).mean()
                    if "content_reliability" in valid
                    else np.nan
                ),
            })
        if include_oracle and "is_malicious" in valid.columns:
            malicious = valid[valid["is_malicious"].astype(str).str.lower().isin({"true", "1", "1.0"})]
            benign = valid[~valid["is_malicious"].astype(str).str.lower().isin({"true", "1", "1.0"})]
            row["oracle_malicious_normalized_weight_mean"] = float(
                pd.to_numeric(malicious["normalized_aggregation_weight"], errors="coerce").mean()
            )
            row["oracle_benign_normalized_weight_mean"] = float(
                pd.to_numeric(benign["normalized_aggregation_weight"], errors="coerce").mean()
            )
            row["oracle_diagnostic_scope"] = "experiment-only; not a deployable defense signal"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--include-oracle", action="store_true")
    args = parser.parse_args()
    files = sorted(Path(args.indir).glob("**/fedagg_teacher_admission_*.csv"))
    if not files:
        raise FileNotFoundError(f"No teacher admission CSV files under {args.indir}")
    result = analyze(
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True),
        include_oracle=bool(args.include_oracle),
    )
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"[write] {output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
