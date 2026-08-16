"""Fail-closed validation for v3 result tables and run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_schema import (
    RESULT_SCHEMA_VERSION,
    VCAA_ALGORITHM_VERSION,
    VCAA_V4_ADMISSION_COLUMNS,
    VCAA_V4_RUNTIME_COLUMNS,
    validate_frame,
)


_TABLE_PREFIXES = {
    "round": "fedagg_experiment_results_",
    "summary": "fedagg_run_summary_",
    "admission": "fedagg_teacher_admission_",
    "defense": "fedagg_teacher_defense_",
    "runtime": "fedagg_runtime_events_",
    "backdoor": "fedagg_backdoor_defense_",
}

# These tables intentionally have narrower schemas than the round/summary
# result tables.  They are validated by their own lineage and identity
# columns instead of being passed to result_schema.validate_frame(), whose
# required columns describe a round-level result row.
_AUXILIARY_REQUIRED_COLUMNS = {
    "admission": {
        "run_uid",
        "dataset",
        "seed",
        "round",
        "admission_method",
        "client_id",
        "admitted",
        "vcaa_algorithm_version",
        "result_schema_version",
    },
    "defense": {
        "run_uid",
        "dataset",
        "seed",
        "round",
        "defense_method",
        "client_id",
        "niabd_algorithm_version",
        "result_schema_version",
    },
    "runtime": {
        "run_uid",
        "dataset",
        "seed",
        "runtime",
        "strategy",
        "task_id",
        "packet_id",
        "result_schema_version",
        "vcaa_algorithm_version",
        "aggregation_algorithm_version",
        "run_class",
        "attack_condition",
    },
    "backdoor": {
        "run_uid",
        "dataset",
        "seed",
        "round",
        "runtime",
        "strategy",
        "attack_type",
        "attack_plan_id",
        "client_id",
        "attack_active",
        "diagnostic_scope",
        "diagnostic_usage",
    },
}


def _table_kind(path: Path) -> str | None:
    for kind, prefix in _TABLE_PREFIXES.items():
        if path.name.startswith(prefix) and path.name.endswith(".csv"):
            return kind
    return None


def _validate_auxiliary_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    kind: str,
) -> None:
    required = _AUXILIARY_REQUIRED_COLUMNS[kind]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path.name} ({kind}) missing columns: {missing}"
        )
    if frame.empty:
        return
    if frame["run_uid"].isna().any() or (
        frame["run_uid"].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{path.name} contains an empty run_uid.")
    if "result_schema_version" in frame.columns:
        versions = set(frame["result_schema_version"].astype(str))
        if versions != {RESULT_SCHEMA_VERSION}:
            raise ValueError(
                f"{path.name} contains invalid result_schema_version: "
                f"{sorted(versions)}"
            )
    for column in (
        "vcaa_algorithm_version",
        "niabd_algorithm_version",
        "aggregation_algorithm_version",
    ):
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.eq("nan").any():
            raise ValueError(f"{path.name} contains empty {column} values.")
    v4 = frame[
        frame.get("vcaa_algorithm_version", pd.Series(index=frame.index))
        .astype(str)
        .eq(VCAA_ALGORITHM_VERSION)
    ]
    if v4.empty:
        return
    if kind == "admission":
        missing = sorted(VCAA_V4_ADMISSION_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing VCAA v4 admission columns: {missing}"
            )
        _validate_v4_admission_frame(v4, path=path)
    elif kind == "runtime":
        missing = sorted(VCAA_V4_RUNTIME_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path.name} is missing VCAA v4 runtime columns: {missing}"
            )


def _as_bool(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "1.0"})


def _validate_v4_admission_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
) -> None:
    hard_valid = _as_bool(frame["hard_valid"])
    raw = pd.to_numeric(frame["aggregation_weight"], errors="coerce")
    normalized = pd.to_numeric(
        frame["normalized_aggregation_weight"],
        errors="coerce",
    )
    ratio = pd.to_numeric(
        frame["effective_weight_ratio_to_uniform"],
        errors="coerce",
    )
    final_score_used = _as_bool(frame["vcaa_final_score_used_for_weighting"])
    reliability = pd.to_numeric(
        frame["content_reliability"],
        errors="coerce",
    )
    if raw[hard_valid].isna().any() or (raw[hard_valid] <= 0.0).any():
        raise ValueError(f"{path.name} has invalid VCAA v4 raw valid weights.")
    if normalized[hard_valid].isna().any() or (normalized[hard_valid] <= 0.0).any():
        raise ValueError(
            f"{path.name} has invalid VCAA v4 normalized valid weights."
        )
    if ratio[hard_valid].isna().any() or (ratio[hard_valid] <= 0.0).any():
        raise ValueError(
            f"{path.name} has invalid VCAA v4 uniform contribution ratios."
        )
    if reliability[hard_valid].isna().any():
        raise ValueError(
            f"{path.name} has missing VCAA v4 content reliability for valid teachers."
        )
    if (reliability[hard_valid] <= 0.0).any() or (reliability[hard_valid] > 1.0).any():
        raise ValueError(
            f"{path.name} has out-of-range VCAA v4 content reliability."
        )
    if final_score_used.any():
        raise ValueError(
            f"{path.name} incorrectly uses the legacy final_score for weighting."
        )
    if raw[~hard_valid].notna().any() and (raw[~hard_valid].fillna(0.0) != 0.0).any():
        raise ValueError(f"{path.name} assigns raw weight to a hard-invalid teacher.")
    if normalized[~hard_valid].notna().any() and (
        normalized[~hard_valid].fillna(0.0) != 0.0
    ).any():
        raise ValueError(
            f"{path.name} assigns normalized weight to a hard-invalid teacher."
        )
    for (run_uid, round_id), group in frame.groupby(
        ["run_uid", "round"], dropna=False
    ):
        valid = _as_bool(group["hard_valid"])
        if not valid.any():
            continue
        normalized_sum = pd.to_numeric(
            group.loc[valid, "normalized_aggregation_weight"],
            errors="coerce",
        ).sum()
        if not math.isclose(float(normalized_sum), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"{path.name} normalized VCAA weights do not sum to one "
                f"for run={run_uid}, round={round_id}."
            )
        ratio_mean = pd.to_numeric(
            group.loc[valid, "effective_weight_ratio_to_uniform"],
            errors="coerce",
        ).mean()
        if not math.isclose(float(ratio_mean), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"{path.name} VCAA uniform contribution ratios are not centered "
                f"at one for run={run_uid}, round={round_id}."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    args = parser.parse_args()
    root = Path(args.indir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("FAIL: run_manifest.json is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if recorded != actual:
        raise SystemExit("FAIL: run manifest hash mismatch.")
    checked = 0
    for path in sorted(root.glob("fedagg_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        kind = _table_kind(path)
        if kind is None:
            raise SystemExit(
                f"FAIL: unsupported v3 result table filename: {path.name}"
            )
        if kind in {"round", "summary"}:
            validate_frame(
                frame,
                require_rounds=(kind == "round"),
            )
        else:
            _validate_auxiliary_frame(frame, path=path, kind=kind)
        checked += 1
    if checked == 0:
        raise SystemExit("FAIL: no non-empty v3 result table found.")
    print(f"PASS: {checked} v3 result table(s), schema={RESULT_SCHEMA_VERSION}.")


if __name__ == "__main__":
    main()
