from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd


RESULT_SCHEMA_VERSION = "fedagg-results-v3"
VCAA_ALGORITHM_VERSION = "vcaa-v4-fresh-first-robust-relative-weighting"
NIABD_ALGORITHM_VERSION = (
    "niabd-v3-trusted-memory-recovery-controller"
)
AGGREGATION_ALGORITHM_VERSION = "aggregation-v1-probability-space"

VCAA_V4_ROUND_COLUMNS = frozenset({
    "vcaa_freshness_valid_teachers",
    "vcaa_effective_teacher_count",
    "vcaa_weight_cv",
    "vcaa_weight_total_variation_from_uniform",
    "vcaa_content_reliability_saturation_fraction",
    "vcaa_content_score_center",
    "vcaa_content_score_scale",
    "vcaa_content_threshold_role",
    "vcaa_threshold_used_for_weighting",
})

VCAA_V4_ADMISSION_COLUMNS = frozenset({
    "content_score_center",
    "content_score_scale",
    "content_score_z",
    "normalized_aggregation_weight",
    "effective_weight_ratio_to_uniform",
    "weighting_mode",
    "vcaa_threshold_used_for_weighting",
    "vcaa_final_score_used_for_weighting",
})

VCAA_V4_RUNTIME_COLUMNS = frozenset({
    "vcaa_content_score_center",
    "vcaa_content_score_scale",
    "vcaa_content_score_z",
    "vcaa_normalized_aggregation_weight",
    "vcaa_effective_weight_ratio_to_uniform",
    "vcaa_weighting_mode",
    "vcaa_threshold_used_for_weighting",
    "vcaa_final_score_used_for_weighting",
})


@dataclass(frozen=True)
class MetricSchemaEntry:
    name: str
    type: str
    nullable: bool
    applicable_methods: tuple[str, ...]
    applicable_runtimes: tuple[str, ...]
    missing_value: object
    description: str
    unit: str = ""


SCHEMA_ENTRIES: tuple[MetricSchemaEntry, ...] = (
    MetricSchemaEntry("run_uid", "string", False, ("*",), ("*",), "", "Stable run identity."),
    MetricSchemaEntry("result_schema_version", "string", False, ("*",), ("*",), RESULT_SCHEMA_VERSION, "Typed result schema lineage."),
    MetricSchemaEntry("vcaa_algorithm_version", "string", False, ("*",), ("*",), "none", "VCAA implementation lineage."),
    MetricSchemaEntry("niabd_algorithm_version", "string", False, ("*",), ("*",), "none", "NIABD implementation lineage."),
    MetricSchemaEntry("aggregation_algorithm_version", "string", False, ("*",), ("*",), AGGREGATION_ALGORITHM_VERSION, "Probability-space aggregation lineage."),
    MetricSchemaEntry("run_class", "string", False, ("*",), ("*",), "", "formal, smoke, synthetic, or control."),
    MetricSchemaEntry("attack_condition", "string", False, ("*",), ("*",), "none", "clean, attacked, or triggered-no-poison."),
    MetricSchemaEntry("transaction_id", "string", False, ("*",), ("*",), "", "Round transaction identity."),
    MetricSchemaEntry("transaction_status", "string", False, ("*",), ("*",), "committed", "prepare/commit/abort status."),
    MetricSchemaEntry("student_snapshot_sha256", "string", True, ("*",), ("*",), None, "Pre-update student proxy logits identity."),
    MetricSchemaEntry("received_teachers", "integer", False, ("*",), ("*",), 0, "Accepted packet count before VCAA."),
    MetricSchemaEntry("admitted_teachers", "integer", False, ("*",), ("*",), 0, "VCAA admitted teacher count."),
    MetricSchemaEntry("memory_candidate_teachers", "integer", True, ("niabd", "vcaa-niabd"), ("*",), None, "NIABD candidate count."),
    MetricSchemaEntry("normal_eligible_teachers", "integer", True, ("niabd", "vcaa-niabd"), ("*",), None, "Normal memory-eligible count."),
    MetricSchemaEntry("drift_recovery_candidates", "integer", True, ("niabd", "vcaa-niabd"), ("*",), None, "Consensus drift recovery count."),
    MetricSchemaEntry("memory_update_teachers", "integer", True, ("niabd", "vcaa-niabd"), ("*",), None, "Raw teachers used for memory update."),
    MetricSchemaEntry("teachers_purified", "integer", False, ("*",), ("*",), 0, "Teacher packets passed through purification."),
    MetricSchemaEntry("niabd_defense_available", "boolean", True, ("niabd", "vcaa-niabd"), ("*",), None, "Whether trusted memory exists."),
    MetricSchemaEntry("niabd_purification_applied", "boolean", True, ("niabd", "vcaa-niabd"), ("*",), None, "Whether purification was executed."),
    MetricSchemaEntry("niabd_memory_updated", "boolean", True, ("niabd", "vcaa-niabd"), ("*",), None, "Whether memory changed this round."),
    MetricSchemaEntry("niabd_memory_update_reason", "string", True, ("niabd", "vcaa-niabd"), ("*",), None, "Explicit NIABD state-machine reason."),
    MetricSchemaEntry("niabd_observations", "number", True, ("niabd", "vcaa-niabd"), ("*",), None, "Cumulative eligible teacher-proxy observations."),
    MetricSchemaEntry("vcaa_history_size", "integer", True, ("vcaa", "vcaa-niabd"), ("*",), None, "VCAA history size."),
    MetricSchemaEntry("numeric_failure_count", "integer", False, ("*",), ("*",), 0, "Actual numeric failures."),
    MetricSchemaEntry("rollback_reason", "string", True, ("*",), ("*",), None, "Round/run rollback reason."),
    MetricSchemaEntry("checkpoint_path", "string", True, ("*",), ("*",), None, "Committed checkpoint path."),
    MetricSchemaEntry("checkpoint_sha256", "string", True, ("*",), ("*",), None, "Committed checkpoint hash."),
    MetricSchemaEntry("git_commit_sha", "string", True, ("*",), ("*",), None, "Git commit used for the run."),
    MetricSchemaEntry("git_dirty", "boolean", True, ("*",), ("*",), None, "Whether the source tree was dirty."),
    MetricSchemaEntry("config_sha256", "string", True, ("*",), ("*",), None, "Run configuration/manifest hash."),
    MetricSchemaEntry("runtime_profile_sha256", "string", True, ("*",), ("*",), None, "Runtime profile hash when applicable."),
)

OPTIONAL_BACKWARD_COMPAT_COLUMNS = frozenset({
    "git_commit_sha",
    "git_dirty",
    "config_sha256",
    "runtime_profile_sha256",
})


def schema_dict() -> dict:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "entries": [asdict(entry) for entry in SCHEMA_ENTRIES],
    }


def schema_hash() -> str:
    payload = json.dumps(schema_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def algorithm_versions(*, vcaa_enabled: bool, niabd_enabled: bool) -> dict[str, str]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "vcaa_algorithm_version": VCAA_ALGORITHM_VERSION if vcaa_enabled else "none",
        "niabd_algorithm_version": NIABD_ALGORITHM_VERSION if niabd_enabled else "none",
        "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
    }


def required_columns() -> set[str]:
    # These fields are appended by new runs but are absent from immutable v3
    # baseline CSVs.  Old frames remain valid and merge with NaN/empty values.
    return {
        entry.name
        for entry in SCHEMA_ENTRIES
        if entry.name not in OPTIONAL_BACKWARD_COMPAT_COLUMNS
    }


def validate_frame(
    frame: pd.DataFrame,
    *,
    method: Optional[str] = None,
    runtime: Optional[str] = None,
    require_rounds: bool = False,
) -> None:
    missing = sorted(required_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"result schema {RESULT_SCHEMA_VERSION} missing columns: {missing}")
    if frame.empty:
        raise ValueError("result schema rejects an empty result frame.")
    if frame["run_uid"].isna().any() or (frame["run_uid"].astype(str).str.strip() == "").any():
        raise ValueError("run_uid must be non-empty for every result row.")
    for column in ("result_schema_version", "vcaa_algorithm_version", "niabd_algorithm_version", "aggregation_algorithm_version"):
        if column not in frame.columns:
            raise ValueError(f"missing algorithm lineage column: {column}")
    if set(frame["result_schema_version"].astype(str)) != {RESULT_SCHEMA_VERSION}:
        raise ValueError("mixed or invalid result_schema_version.")
    for column in (
        "vcaa_algorithm_version",
        "niabd_algorithm_version",
        "aggregation_algorithm_version",
    ):
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.eq("nan").any():
            raise ValueError(f"{column} cannot be empty or NaN.")
    if "transaction_status" in frame.columns:
        statuses = set(frame["transaction_status"].dropna().astype(str))
        if not statuses.issubset({"committed", "aborted"}):
            raise ValueError("invalid transaction_status in result rows.")
    for column in (
        "received_teachers",
        "admitted_teachers",
        "memory_candidate_teachers",
        "normal_eligible_teachers",
        "drift_recovery_candidates",
        "memory_update_teachers",
        "vcaa_history_size",
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if (values < 0).any() or (~(values % 1 == 0)).any():
            raise ValueError(f"{column} must contain non-negative integers.")
    for column in ("checkpoint_sha256", "student_snapshot_sha256"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str)
            if any(value and len(value) != 64 for value in values):
                raise ValueError(f"{column} contains an invalid SHA-256 value.")
    vcaa_v4_rows = frame[
        frame["vcaa_algorithm_version"].astype(str)
        == VCAA_ALGORITHM_VERSION
    ]
    # The v4 diagnostics below are round-level fields.  Summary rows carry
    # the same algorithm lineage but intentionally retain their compact,
    # historical summary schema.
    if not vcaa_v4_rows.empty and "round" in frame.columns:
        missing_v4 = sorted(VCAA_V4_ROUND_COLUMNS - set(frame.columns))
        if missing_v4:
            raise ValueError(
                "VCAA v4 round result is missing columns: "
                f"{missing_v4}"
            )
        valid_counts = pd.to_numeric(
            vcaa_v4_rows["vcaa_freshness_valid_teachers"],
            errors="coerce",
        )
        if valid_counts.isna().any() or (valid_counts < 0).any():
            raise ValueError("VCAA v4 freshness-valid teacher counts are invalid.")
        ess = pd.to_numeric(
            vcaa_v4_rows["vcaa_effective_teacher_count"],
            errors="coerce",
        )
        cv = pd.to_numeric(vcaa_v4_rows["vcaa_weight_cv"], errors="coerce")
        tv = pd.to_numeric(
            vcaa_v4_rows["vcaa_weight_total_variation_from_uniform"],
            errors="coerce",
        )
        saturation = pd.to_numeric(
            vcaa_v4_rows["vcaa_content_reliability_saturation_fraction"],
            errors="coerce",
        )
        observed = valid_counts > 0
        for name, values in (
            ("ESS", ess),
            ("weight CV", cv),
            ("weight TV", tv),
            ("saturation", saturation),
        ):
            observed_values = values[observed]
            if observed_values.isna().any() or not observed_values.map(
                lambda value: math.isfinite(float(value))
            ).all():
                raise ValueError(
                    f"VCAA v4 {name} diagnostics must be finite."
                )
        if (ess[observed] < 1.0).any():
            raise ValueError("VCAA v4 ESS must be at least one.")
        if (ess[observed] > valid_counts[observed] + 1e-5).any():
            raise ValueError("VCAA v4 ESS exceeds freshness-valid teacher count.")
        if (cv[observed] < 0.0).any() or (tv[observed] < 0.0).any():
            raise ValueError("VCAA v4 weight dispersion diagnostics are negative.")
        if (saturation[observed] < 0.0).any() or (saturation[observed] > 1.0).any():
            raise ValueError("VCAA v4 saturation fraction is outside [0, 1].")
    if method is not None:
        method = str(method).lower()
        expected_vcaa = VCAA_ALGORITHM_VERSION if method in {"vcaa", "vcaa-niabd"} else "none"
        expected_niabd = NIABD_ALGORITHM_VERSION if method in {"niabd", "vcaa-niabd"} else "none"
        if set(frame["vcaa_algorithm_version"].astype(str)) != {expected_vcaa}:
            raise ValueError("VCAA algorithm lineage does not match method.")
        if set(frame["niabd_algorithm_version"].astype(str)) != {expected_niabd}:
            raise ValueError("NIABD algorithm lineage does not match method.")
    if runtime is not None and "runtime" in frame.columns:
        if set(frame["runtime"].astype(str)) != {str(runtime)}:
            raise ValueError("result runtime lineage does not match expected runtime.")
    if require_rounds and "round" in frame.columns:
        grouped = frame.groupby("run_uid", dropna=False)
        for run_uid, group in grouped:
            rounds = pd.to_numeric(group["round"], errors="coerce").dropna().astype(int).sort_values().tolist()
            if rounds != list(range(1, len(rounds) + 1)):
                raise ValueError(f"rounds are missing or duplicated for run_uid={run_uid}.")


def write_schema(path: str | Path) -> None:
    Path(path).write_text(json.dumps(schema_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
