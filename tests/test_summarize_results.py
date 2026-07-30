import pandas as pd
import pytest

from summarize_results import summarize


def test_summary_aggregates_seed_runs(tmp_path):
    rows = [
        {
            "run_uid": "a",
            "dataset": "cifar10",
            "seed": 0,
            "runtime": "sync",
            "strategy": "baseline",
            "topology": "server-client",
            "server_model": "resnet18",
            "client_model": "resnet18",
            "server_device": "cpu",
            "client_device": "cpu",
            "num_clients": 2,
            "partition_scheme": "iid",
            "knowledge_interface": "serialized-proxy-logits",
            "aggregation_rule": "mean-soft-probabilities",
            "vcaa_enabled": 0,
            "admission_method": "none",
            "niabd_enabled": 0,
            "defense_method": "none",
            "rounds": 1,
            "final_accuracy": 0.6,
            "best_accuracy": 0.6,
            "final_loss": 0.9,
            "wall_clock_time_s": 1.0,
            "total_rollbacks": 0,
            "total_numeric_failures": 0,
            "mean_teacher_utilization": 1.0,
            "total_teachers_admitted": 2,
            "total_teachers_rejected": 0,
            "total_client_upload_bytes": 80,
            "total_server_broadcast_bytes": 80,
            "total_teachers_purified": 0,
            "mean_niabd_anomaly_fraction": 0.0,
            "mean_niabd_suppression": 0.0,
            "final_niabd_threshold_mean": 0.0,
            "total_niabd_prototype_updates": 0,
            "total_client_wire_bytes": 0,
            "total_packets_consumed": 0,
            "total_stale_packets": 0,
            "max_version_lag": 0,
            "stale_rejection_rate": float("nan"),
            "fresh_rejection_rate": float("nan"),
        },
        {
            "run_uid": "b",
            "dataset": "cifar10",
            "seed": 1,
            "runtime": "sync",
            "strategy": "baseline",
            "topology": "server-client",
            "server_model": "resnet18",
            "client_model": "resnet18",
            "server_device": "cpu",
            "client_device": "cpu",
            "num_clients": 2,
            "partition_scheme": "iid",
            "knowledge_interface": "serialized-proxy-logits",
            "aggregation_rule": "mean-soft-probabilities",
            "vcaa_enabled": 0,
            "admission_method": "none",
            "niabd_enabled": 0,
            "defense_method": "none",
            "rounds": 1,
            "final_accuracy": 0.8,
            "best_accuracy": 0.8,
            "final_loss": 0.7,
            "wall_clock_time_s": 1.2,
            "total_rollbacks": 0,
            "total_numeric_failures": 0,
            "mean_teacher_utilization": 1.0,
            "total_teachers_admitted": 2,
            "total_teachers_rejected": 0,
            "total_client_upload_bytes": 80,
            "total_server_broadcast_bytes": 80,
            "total_teachers_purified": 0,
            "mean_niabd_anomaly_fraction": 0.0,
            "mean_niabd_suppression": 0.0,
            "final_niabd_threshold_mean": 0.0,
            "total_niabd_prototype_updates": 0,
            "total_client_wire_bytes": 0,
            "total_packets_consumed": 0,
            "total_stale_packets": 0,
            "max_version_lag": 0,
            "stale_rejection_rate": float("nan"),
            "fresh_rejection_rate": float("nan"),
        },
    ]
    pd.DataFrame(rows).to_csv(
        tmp_path / "fedagg_run_summary_cifar10.csv",
        index=False,
    )

    result = summarize(str(tmp_path))

    assert len(result) == 1
    assert result.loc[0, "runs"] == 2
    assert result.loc[0, "final_accuracy_mean"] == pytest.approx(0.7)


def test_summary_does_not_turn_unobserved_process_metrics_into_zero(tmp_path):
    source = pd.DataFrame([{
        column: value
        for column, value in rows_for_missing_process_summary().items()
    }])
    source.to_csv(
        tmp_path / "fedagg_run_summary_cifar10.csv",
        index=False,
    )

    result = summarize(str(tmp_path))

    assert pd.isna(result.loc[0, "total_client_wire_bytes_mean"])
    assert pd.isna(result.loc[0, "total_client_wire_bytes_std"])
    assert pd.isna(result.loc[0, "total_packets_consumed_mean"])
    assert pd.isna(result.loc[0, "total_stale_packets_mean"])
    assert pd.isna(result.loc[0, "max_version_lag_mean"])


def rows_for_missing_process_summary():
    return {
        "run_uid": "sync-missing",
        "dataset": "cifar10",
        "seed": 0,
        "runtime": "sync",
        "strategy": "baseline",
        "topology": "server-client",
        "server_model": "resnet18",
        "client_model": "resnet18",
        "server_device": "cpu",
        "client_device": "cpu",
        "num_clients": 2,
        "partition_scheme": "iid",
        "knowledge_interface": "serialized-proxy-logits",
        "aggregation_rule": "mean-soft-probabilities",
        "vcaa_enabled": 0,
        "admission_method": "none",
        "niabd_enabled": 0,
        "defense_method": "none",
        "final_accuracy": 0.5,
        "best_accuracy": 0.5,
        "final_loss": 1.0,
        "wall_clock_time_s": 1.0,
        "total_rollbacks": 0,
        "total_numeric_failures": 0,
        "mean_teacher_utilization": 1.0,
        "total_teachers_admitted": 2,
        "total_teachers_rejected": 0,
        "total_client_upload_bytes": 10,
        "total_server_broadcast_bytes": 10,
        "total_teachers_purified": 0,
        "mean_niabd_anomaly_fraction": float("nan"),
        "mean_niabd_suppression": float("nan"),
        "final_niabd_threshold_mean": float("nan"),
        "total_niabd_prototype_updates": 0,
        "total_client_wire_bytes": float("nan"),
        "total_packets_consumed": float("nan"),
        "total_stale_packets": float("nan"),
        "max_version_lag": float("nan"),
        "stale_rejection_rate": float("nan"),
        "fresh_rejection_rate": float("nan"),
    }
