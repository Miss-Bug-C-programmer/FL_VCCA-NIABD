from pathlib import Path

import pandas as pd
import pytest

from result_schema import (
    AGGREGATION_ALGORITHM_VERSION,
    RESULT_SCHEMA_VERSION,
    NIABD_ALGORITHM_VERSION,
    VCAA_ALGORITHM_VERSION,
)
from scripts.validate_v3_results import (
    _table_kind,
    _validate_auxiliary_frame,
)

LEGACY_VCAA_ALGORITHM_VERSION = "vcaa-v3-absolute-freshness-robust-content"


def test_v3_validator_classifies_all_exported_tables():
    assert _table_kind(Path("fedagg_experiment_results_cifar10.csv")) == "round"
    assert _table_kind(Path("fedagg_run_summary_cifar10.csv")) == "summary"
    assert _table_kind(Path("fedagg_teacher_admission_cifar10.csv")) == "admission"
    assert _table_kind(Path("fedagg_teacher_defense_cifar10.csv")) == "defense"
    assert _table_kind(Path("fedagg_runtime_events_cifar10.csv")) == "runtime"
    assert _table_kind(Path("fedagg_backdoor_defense_cifar10.csv")) == "backdoor"
    assert _table_kind(Path("unknown.csv")) is None


@pytest.mark.parametrize(
    ("kind", "filename", "extra"),
    [
        (
            "admission",
            "fedagg_teacher_admission_cifar10.csv",
            {
                "round": 1,
                "admission_method": "vcaa",
                "client_id": 0,
                "admitted": True,
                    "vcaa_algorithm_version": LEGACY_VCAA_ALGORITHM_VERSION,
                "result_schema_version": RESULT_SCHEMA_VERSION,
            },
        ),
        (
            "defense",
            "fedagg_teacher_defense_cifar10.csv",
            {
                "round": 1,
                "defense_method": "niabd",
                "client_id": 0,
                "niabd_algorithm_version": NIABD_ALGORITHM_VERSION,
                "result_schema_version": RESULT_SCHEMA_VERSION,
            },
        ),
        (
            "runtime",
            "fedagg_runtime_events_cifar10.csv",
            {
                "runtime": "process-semi-async",
                "strategy": "vcaa-niabd",
                "task_id": "task-1",
                "packet_id": "packet-1",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                    "vcaa_algorithm_version": LEGACY_VCAA_ALGORITHM_VERSION,
                "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
                "run_class": "formal",
                "attack_condition": "attacked",
            },
        ),
        (
            "backdoor",
            "fedagg_backdoor_defense_cifar10.csv",
            {
                "round": 1,
                "runtime": "sync",
                "strategy": "baseline",
                "attack_type": "badnets",
                "attack_plan_id": "plan-1",
                "client_id": 0,
                "attack_active": True,
                "diagnostic_scope": "experiment-only oracle diagnostic",
                "diagnostic_usage": "not a deployable defense signal",
            },
        ),
    ],
)
def test_v3_validator_accepts_narrow_auxiliary_tables(kind, filename, extra):
    row = {
        "run_uid": "run-1",
        "dataset": "cifar10",
        "seed": 0,
        **extra,
    }
    _validate_auxiliary_frame(
        pd.DataFrame([row]),
        path=Path(filename),
        kind=kind,
    )


def test_v5_admission_validator_checks_normalized_contribution_fields():
    row = {
        "run_uid": "run-v5",
        "dataset": "cifar10",
        "seed": 0,
        "round": 1,
        "admission_method": "vcaa",
        "client_id": 0,
        "admitted": True,
        "hard_valid": True,
        "content_valid": True,
        "content_gate_active": True,
        "content_threshold": 0.4,
        "content_threshold_source": "historical_median_minus_mad_floor",
        "content_history_observations": 3,
        "content_rejection_reason": "",
        "rejection_reason": "",
        "timestamp_valid": True,
        "version_lag": 0,
        "knowledge_age_s": 0.0,
        "generated_at_s": 0.0,
        "received_at_s": 0.1,
        "consumed_at_s": 0.2,
        "version_lag_score": 1.0,
        "age_score": 1.0,
        "raw_version_lag": 0,
        "effective_age_half_life_s": 1.0,
        "effective_max_knowledge_age_s": 4.0,
        "age_scale_mode": "fixed",
        "aggregation_weight": 1.0,
        "content_reliability": 0.5,
        "normalized_aggregation_weight": 1.0,
        "effective_weight_ratio_to_uniform": 1.0,
        "content_score_center": 0.5,
        "content_score_scale": 0.05,
        "content_score_z": 0.0,
        "weighting_mode": "robust_relative_sigmoid",
        "vcaa_threshold_used_for_weighting": True,
        "vcaa_final_score_used_for_weighting": True,
        "vcaa_algorithm_version": VCAA_ALGORITHM_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }
    _validate_auxiliary_frame(
        pd.DataFrame([row]),
        path=Path("fedagg_teacher_admission_cifar10.csv"),
        kind="admission",
    )
