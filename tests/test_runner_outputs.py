from experiment_runner import (
    _admission_rows,
    _defense_rows,
    _round_rows,
    _runtime_event_rows,
    _summary_row,
)


def _metrics():
    return {
        "acc_list": [0.4, 0.6],
        "loss_list": [1.2, 0.8],
        "local_train_time_s": [1.0, 1.1],
        "upload_time_s": [0.2, 0.3],
        "admission_time_s": [0.1, 0.1],
        "defense_time_s": [0.1, 0.2],
        "distill_time_s": [0.5, 0.6],
        "round_time_s": [1.5, 1.7],
        "wall_clock_time_s": [1.5, 3.2],
        "clients_trained": [3, 3],
        "server_client_distillations": [3, 3],
        "server_updates_from_clients": [3, 2],
        "client_upload_bytes": [120, 120],
        "server_broadcast_bytes": [120, 120],
        "client_reverse_distillations": [3, 3],
        "server_update_applied": [1, 1],
        "teachers_admitted": [3, 2],
        "teachers_rejected": [0, 1],
        "teacher_utilization": [1.0, 2.0 / 3.0],
        "admission_threshold": [0.0, 0.5],
        "admission_score_mean": [0.8, 0.7],
        "vcaa_version_score_mean": [1.0, 0.9],
        "vcaa_content_score_mean": [0.6, 0.5],
        "vcaa_proxy_accuracy_mean": [0.7, 0.6],
        "vcaa_entropy_mean": [0.4, 0.5],
        "vcaa_kl_mean": [0.1, 0.2],
        "vcaa_enabled": 1,
        "admission_method": "vcaa",
        "niabd_enabled": 1,
        "defense_method": "niabd",
        "teachers_purified": [3, 2],
        "niabd_warmup": [1.0, 0.0],
        "niabd_anomaly_fraction": [0.0, 0.1],
        "niabd_mean_suppression": [0.0, 0.05],
        "niabd_threshold_mean": [2.0, 2.1],
        "niabd_threshold_min": [2.0, 2.0],
        "niabd_threshold_max": [2.0, 2.2],
        "niabd_prototype_updated": [1.0, 1.0],
        "niabd_prototype_observations": [12.0, 20.0],
        "niabd_memory_eligible_teachers": [3, 1],
        "knowledge_interface": "serialized-proxy-logits",
        "aggregation_rule": "mean-soft-probabilities",
        "teacher_admission_records": [
            [
                {
                    "client_id": 0,
                    "admitted": True,
                    "score": 0.8,
                    "version_score": 1.0,
                }
            ],
            [
                {
                    "client_id": 0,
                    "admitted": False,
                    "score": 0.4,
                    "version_score": 0.8,
                }
            ],
        ],
        "teacher_defense_records": [
            [
                {
                    "client_id": 0,
                    "anomaly_fraction": 0.0,
                    "mean_abs_deviation": 0.0,
                    "max_abs_deviation": 0.0,
                    "mean_suppression": 0.0,
                    "memory_eligible": True,
                }
            ],
            [
                {
                    "client_id": 0,
                    "anomaly_fraction": 0.2,
                    "mean_abs_deviation": 1.0,
                    "max_abs_deviation": 5.0,
                    "mean_suppression": 0.1,
                    "memory_eligible": False,
                }
            ],
        ],
        "nonfinite_eval_batches": [0, 0],
        "nonfinite_distill_rollbacks": [0, 0],
        "numeric_failure_count": [0.0, 0.0],
    }


def test_round_rows_use_server_client_roles_and_resnet18():
    rows = list(
        _round_rows(
            _metrics(),
            run_uid="abc",
            dataset_name="cifar10",
            seed=0,
            num_clients=3,
            partition_scheme="iid",
        )
    )

    assert len(rows) == 2
    assert rows[0]["topology"] == "server-client"
    assert rows[0]["server_role"] == "global-student"
    assert rows[0]["client_role"] == "local-teacher"
    assert rows[0]["server_model"] == "resnet18"
    assert rows[0]["client_model"] == "resnet18"
    assert rows[0]["vcaa_enabled"] == 1
    assert rows[0]["admission_method"] == "vcaa"
    assert rows[0]["defense_method"] == "niabd"
    assert rows[0]["knowledge_interface"] == "serialized-proxy-logits"
    assert rows[0]["aggregation_rule"] == "mean-soft-probabilities"
    assert rows[0]["client_upload_bytes"] == 120
    assert rows[1]["teachers_rejected"] == 1
    assert "num_edges" not in rows[0]


def test_run_summary_preserves_best_and_final_metrics():
    rows = list(
        _round_rows(
            _metrics(),
            run_uid="abc",
            dataset_name="cifar10",
            seed=0,
            num_clients=3,
            partition_scheme="iid",
        )
    )
    summary = _summary_row(rows)

    assert summary["rounds"] == 2
    assert summary["final_accuracy"] == 0.6
    assert summary["best_accuracy"] == 0.6
    assert summary["topology"] == "server-client"
    assert summary["admission_method"] == "vcaa"
    assert summary["total_teachers_rejected"] == 1
    assert summary["total_teachers_purified"] == 5
    assert summary["defense_method"] == "niabd"
    assert summary["total_client_upload_bytes"] == 240


def test_client_level_admission_records_are_exportable():
    rows = list(
        _admission_rows(
            _metrics(),
            run_uid="abc",
            dataset_name="cifar10",
            seed=0,
            num_clients=3,
            partition_scheme="iid",
        )
    )

    assert len(rows) == 2
    assert rows[0]["admission_method"] == "vcaa"
    assert rows[0]["client_id"] == 0
    assert rows[1]["admitted"] is False


def test_client_level_defense_records_are_exportable():
    rows = list(
        _defense_rows(
            _metrics(),
            run_uid="abc",
            dataset_name="cifar10",
            seed=0,
            num_clients=3,
            partition_scheme="iid",
        )
    )

    assert len(rows) == 2
    assert rows[0]["defense_method"] == "niabd"
    assert rows[0]["memory_eligible"] is True
    assert rows[1]["anomaly_fraction"] == 0.2


def test_packet_runtime_events_keep_run_and_strategy_lineage():
    metrics = _metrics()
    metrics.update({
        "runtime": "process-semi-async",
        "strategy": "vcaa-niabd",
        "server_device": "cuda",
        "client_device": "cpu",
        "runtime_events": [{
            "client_id": 2,
            "task_id": "task-x",
            "packet_id": "packet-x",
            "source_round": 2,
            "consumed_round": 4,
            "version_lag": 2,
            "payload_sha256": "abc",
        }],
    })

    rows = list(
        _runtime_event_rows(
            metrics,
            run_uid="run-x",
            dataset_name="cifar10",
            seed=0,
            num_clients=3,
            partition_scheme="iid",
        )
    )

    assert rows[0]["run_uid"] == "run-x"
    assert rows[0]["strategy"] == "vcaa-niabd"
    assert rows[0]["runtime"] == "process-semi-async"
    assert rows[0]["version_lag"] == 2
