"""Fail-closed validation for v3 result tables and run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_schema import RESULT_SCHEMA_VERSION, validate_frame


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
